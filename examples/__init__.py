"""Example workflows built on agentforge.

Each example exposes a ``build_<name>_workflow()`` function returning a
list of :class:`agentforge.Step` objects. :func:`build_all` collects them
for the CLI (``agentforge run <name>``).
"""

from agentforge.engine import Step

from examples.research_agent import build_research_workflow
from examples.code_review_agent import build_code_review_workflow

__all__ = ["build_all", "build_research_workflow", "build_code_review_workflow"]


def build_all() -> dict[str, list[Step]]:
    return {
        "research": build_research_workflow(),
        "code-review": build_code_review_workflow(),
    }
