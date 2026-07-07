from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd

from simulator.agents_graphs.event_graph import EventGraph


def make_graph(executors=None, restriction_llm=None, final_llm=None):
    executors = executors or {}
    if restriction_llm is None:
        restriction_llm = Mock()
        restriction_llm.invoke.return_value = SimpleNamespace(content="row must be filtered")
    if final_llm is None:
        final_llm = Mock()
        final_llm.invoke.return_value = final_llm
        final_llm.dict.return_value = {"scenario": "final scenario"}
    return EventGraph(executors=executors, llm_filter_constraints=restriction_llm, llm_final_response=final_llm)


def test_get_end_condition_routes_to_executor_until_rows_empty():
    """When rows remain to generate, then the event graph routes to the executor; otherwise it finalizes."""
    condition = make_graph().get_end_condition()

    assert condition({"rows_to_generate": [{"table_name": "users", "row": "{}"}]}) == "executor"
    assert condition({"rows_to_generate": []}) == "final_response_node"


def test_restriction_node_builds_cur_restrictions():
    """When the restriction node runs, then it computes row-level constraints from the current state."""
    restriction_llm = Mock()
    restriction_llm.invoke.return_value = SimpleNamespace(content="filter this row")
    graph = make_graph(restriction_llm=restriction_llm)
    node = graph.get_restriction_node()

    result = node(
        {
            "rows_to_generate": [{"table_name": "users", "row": "name: alice"}],
            "all_restrictions": "max 1 row",
            "variables_definitions": "value: 1\n",
        }
    )

    assert result == {"cur_restrictions": "filter this row"}
    called = restriction_llm.invoke.call_args.args[0]
    assert called["row"] == "name: alice"
    assert "max 1 row" in called["restrictions"]


def test_executor_node_updates_dataset_and_variables():
    """When the executor processes a row, then it updates the dataset, generated rows, and variable definitions."""
    executor = Mock()
    executor.system_prompt = SimpleNamespace(format_messages=Mock(return_value=[SimpleNamespace(content="system")]))
    executor.invoke.return_value = {
        "messages": [SimpleNamespace(content="start"), SimpleNamespace(content="```yml\nvalue: 7\n```")],
        "args": {"dataset": {"users": pd.DataFrame([{"id": 1}])}},
    }
    graph = make_graph(executors={"users": executor})
    node = graph.get_executor_node()
    dataset = {"users": pd.DataFrame([{"id": 1}])}
    state = {
        "rows_to_generate": [{"table_name": "users", "row": "id: 2"}],
        "rows_generated": [],
        "cur_restrictions": None,
        "all_restrictions": "row must exist",
        "variables_definitions": "",
        "dataset": dataset,
    }

    result = node(state)

    assert result["rows_to_generate"] == []
    assert result["rows_generated"] == [{"table_name": "users", "row": "id: 2"}]
    assert "value: 7" in result["variables_definitions"]
    assert list(result["dataset"]["users"].columns) == ["id"]
    called = executor.invoke.call_args.args[0]
    assert called["args"]["dataset"] is dataset


def test_final_node_formats_rows_and_response():
    """When the final node runs, then it formats dataset rows and returns the normalized scenario."""
    final_llm = Mock()
    final_llm.invoke.return_value = final_llm
    final_llm.dict.return_value = {"scenario": "normalized scenario"}
    graph = make_graph(final_llm=final_llm)
    node = graph.get_final_node()
    dataset = {"users": pd.DataFrame([{"id": 1, "name": "alice"}])}

    result = node(
        {
            "dataset": dataset,
            "variables_definitions": "value: 1\n",
            "event_description": "a user update",
        }
    )

    assert result["final_response_scenario"] == "normalized scenario"
    assert "## Table: users" in result["final_response_table_rows"]
    assert final_llm.invoke.call_args.args[0]["scenario"] == "a user update"
