"""Tests for the agentforge CLI helpers."""

from agentforge.cli import format_run_row


def test_format_run_row_pads_name_and_shows_status():
    row = format_run_row({"id": 7, "name": "research", "status": "completed"})
    assert row.startswith("7  research")
    assert row.endswith("completed")


def test_format_run_row_handles_long_names():
    row = format_run_row({"id": 12, "name": "a-very-long-workflow-name", "status": "paused"})
    assert "a-very-long-workflow-name" in row
    assert row.endswith("paused")
