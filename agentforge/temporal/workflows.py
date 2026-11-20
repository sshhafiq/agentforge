"""The durable agent-run workflow.

``AgentRunWorkflow`` is the Temporal-native counterpart of
:class:`agentforge.engine.WorkflowEngine`:

- the plan comes from a planning activity,
- each tool call is an activity with an explicit retry policy (the
  engine's per-step backoff, expressed declaratively),
- tools listed in ``approval_tools`` pause the run until a human sends
  the ``approve`` signal — the engine's HITL gate, except the wait is
  durable: the workflow can sit paused for days at no compute cost,
- live progress is exposed through the ``run_state`` query,
- every transition is mirrored into the :class:`RunStore` by activities,
  so the store remains the system of record for dashboards and audits.

The workflow code is deterministic and side-effect free; all I/O lives in
:mod:`agentforge.temporal.activities`.
"""

from __future__ import annotations

from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError

with workflow.unsafe.imports_passed_through():
    from agentforge.temporal.activities import AgentActivities
    from agentforge.temporal.models import (
        AgentRunInput,
        AgentRunResult,
        ApprovalSignalPayload,
        ExecuteToolInput,
        FinishRunInput,
        PlanStepsInput,
        RecordRunStartedInput,
        RequestApprovalInput,
        ResolveApprovalInput,
        RunState,
        StepRecord,
    )

_BOOKKEEPING_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
)

#: Tool calls get a longer leash than bookkeeping: flaky external tools
#: are exactly what per-step retries exist for.
_TOOL_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=3,
)


@workflow.defn
class AgentRunWorkflow:
    """Plan -> (approve) -> execute, durably."""

    def __init__(self) -> None:
        self._run_id = ""
        self._status = "planning"
        self._total_steps = 0
        self._completed_steps = 0
        self._current_tool: str | None = None
        self._pending_question: str | None = None
        self._decision: ApprovalSignalPayload | None = None

    # ------------------------------------------------------------------
    # signal + query handlers
    # ------------------------------------------------------------------
    @workflow.signal
    def approve(self, payload: ApprovalSignalPayload) -> None:
        """Deliver the human's decision for the pending approval gate."""
        self._decision = payload

    @workflow.query
    def run_state(self) -> RunState:
        return RunState(
            run_id=self._run_id,
            status=self._status,
            total_steps=self._total_steps,
            completed_steps=self._completed_steps,
            current_tool=self._current_tool,
            awaiting_approval=self._status == "awaiting_approval",
            pending_question=self._pending_question,
        )

    # ------------------------------------------------------------------
    # main flow
    # ------------------------------------------------------------------
    @workflow.run
    async def run(self, input: AgentRunInput) -> AgentRunResult:
        self._run_id = input.run_id
        await workflow.execute_activity(
            AgentActivities.record_run_started,
            RecordRunStartedInput(
                run_id=input.run_id, name=input.run_name, task=input.task
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_BOOKKEEPING_RETRY,
        )

        plan = await workflow.execute_activity(
            AgentActivities.plan_steps,
            PlanStepsInput(
                run_id=input.run_id, task=input.task, max_steps=input.max_steps
            ),
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=_BOOKKEEPING_RETRY,
        )
        plan = plan[: input.max_steps]
        self._total_steps = len(plan)
        self._status = "running"

        records: list[StepRecord] = []
        last_observation = ""
        for index, step in enumerate(plan):
            self._current_tool = step.tool
            arguments = dict(step.arguments)
            status = "completed"

            if step.tool in input.approval_tools:
                outcome = await self._approval_gate(
                    input.run_id, index, step.tool, arguments
                )
                if outcome is None:  # rejected
                    record = StepRecord(
                        index=index, tool=step.tool, arguments=arguments,
                        status="rejected",
                        error="rejected by human reviewer",
                    )
                    records.append(record)
                    return await self._finish(
                        input.run_id, "rejected", records,
                        answer="Run rejected by human reviewer.",
                    )
                arguments, edited = outcome
                if edited:
                    status = "edited"

            try:
                result = await workflow.execute_activity(
                    AgentActivities.execute_tool,
                    ExecuteToolInput(
                        run_id=input.run_id, step_index=index,
                        tool=step.tool, arguments=arguments,
                    ),
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=_TOOL_RETRY,
                )
            except ActivityError as exc:
                record = StepRecord(
                    index=index, tool=step.tool, arguments=arguments,
                    status="failed", error=str(exc.cause or exc),
                )
                records.append(record)
                return await self._finish(
                    input.run_id, "failed", records,
                    answer=f"Step {index} ('{step.tool}') failed: {record.error}",
                )

            last_observation = result.observation
            self._completed_steps += 1
            records.append(
                StepRecord(
                    index=index, tool=step.tool, arguments=arguments,
                    status=status, observation=result.observation,
                )
            )

        self._current_tool = None
        answer = last_observation or "Run completed with no tool calls."
        return await self._finish(input.run_id, "completed", records, answer)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    async def _approval_gate(
        self, run_id: str, index: int, tool: str, arguments: dict
    ) -> tuple[dict, bool] | None:
        """Pause until the ``approve`` signal arrives.

        Returns ``(arguments, edited)`` to continue, or ``None`` when the
        reviewer rejected the step.
        """
        approval_id = await workflow.execute_activity(
            AgentActivities.request_approval,
            RequestApprovalInput(
                run_id=run_id, step_index=index, tool=tool, arguments=arguments
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_BOOKKEEPING_RETRY,
        )
        self._status = "awaiting_approval"
        self._pending_question = f"Approve tool call '{tool}'?"
        self._decision = None
        await workflow.wait_condition(lambda: self._decision is not None)
        decision = self._decision
        self._decision = None
        self._status = "running"
        self._pending_question = None

        await workflow.execute_activity(
            AgentActivities.resolve_approval,
            ResolveApprovalInput(
                run_id=run_id, approval_id=approval_id,
                decision=decision.to_decision(),
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_BOOKKEEPING_RETRY,
        )
        if decision.verdict == "reject":
            return None
        if decision.verdict == "edit":
            merged = dict(arguments)
            merged.update(decision.edits)
            return merged, True
        return arguments, False

    async def _finish(
        self, run_id: str, status: str,
        records: list[StepRecord], answer: str,
    ) -> AgentRunResult:
        self._status = "done"
        await workflow.execute_activity(
            AgentActivities.finish_run,
            FinishRunInput(
                run_id=run_id, status=status, answer=answer,
                steps_completed=self._completed_steps,
            ),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=_BOOKKEEPING_RETRY,
        )
        return AgentRunResult(
            run_id=run_id, status=status, steps=records, answer=answer
        )
