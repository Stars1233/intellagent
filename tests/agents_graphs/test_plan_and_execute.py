from unittest.mock import Mock

from langgraph.graph import END

from simulator.agents_graphs.plan_and_execute import Plan, PlanExecuteImplementation, SingleStep, should_end


def make_impl(planner_output, replanner_output, executor=None):
    planner = Mock()
    planner.invoke.return_value = planner_output
    replanner = Mock()
    replanner.invoke.return_value = replanner_output
    return PlanExecuteImplementation(planner=planner, executor=executor or {}, replanner=replanner), planner, replanner


def test_should_end_returns_end_for_empty_plan():
    """When the plan is empty, then the workflow terminates."""
    assert should_end({"plan": []}) == END


def test_should_end_returns_agent_for_non_empty_plan():
    """When plan steps remain, then the workflow routes back to the agent node."""
    assert should_end({"plan": [{"content": "do something"}]}) == "agent"


def test_get_planner_function_maps_plan_shape():
    """When the planner runs, then its structured output is mapped into graph state."""
    planner_output = Plan(
        steps=[SingleStep(content="step 1", executor="search")],
        final_response="done",
    )
    impl, planner, _ = make_impl(planner_output, planner_output)

    result = impl.get_planner_function()({"input": "build a plan"})

    assert result == {
        "plan": planner_output.dict()["steps"],
        "response": "done",
    }
    planner.invoke.assert_called_once_with("build a plan")


def test_get_replanner_function_maps_plan_shape():
    """When replanning runs, then the returned plan and final response are preserved in graph state."""
    replanner_output = Plan(
        steps=[SingleStep(content="step 2", executor="Response")],
        final_response="final answer",
    )
    impl, _, replanner = make_impl(replanner_output, replanner_output)

    state = {"input": "prompt", "plan": [], "past_steps": [], "response": "", "args": {}}
    result = impl.get_replanner_function()(state)

    assert result == {
        "plan": replanner_output.dict()["steps"],
        "response": "final answer",
    }
    replanner.invoke.assert_called_once_with(state)


def test_executor_short_circuits_for_response_step():
    """When the next step is a direct response, then the executor short-circuits without calling tools."""
    planner_output = Plan(steps=[SingleStep(content="step 1", executor="Response")], final_response="done")
    impl, _, _ = make_impl(planner_output, planner_output)

    result = impl.get_executor_function()(
        {"plan": [{"content": "answer the user", "executor": "Response"}], "args": {"x": 1}}
    )

    assert result == {"past_steps": [("answer the user", "answer the user")]}


def test_executor_dispatches_to_named_executor():
    """When the next step targets a named executor, then the implementation invokes that executor with args."""
    named_executor = Mock()
    named_executor.invoke.return_value = {
        "messages": [Mock(content="intermediate"), Mock(content="final tool result")],
        "args": {"updated": True},
    }
    planner_output = Plan(steps=[SingleStep(content="step 1", executor="search")], final_response="done")
    impl, _, _ = make_impl(planner_output, planner_output, executor={"search": named_executor})

    result = impl.get_executor_function()(
        {"plan": [{"content": "look up account", "executor": "search"}], "args": {"session": "abc"}}
    )

    assert result == {
        "past_steps": [("look up account", "final tool result")],
        "args": {"updated": True},
    }
    assert "look up account" in named_executor.invoke.call_args.args[0]
    assert named_executor.invoke.call_args.kwargs == {"additional_args": {"session": "abc"}}


def test_graph_smoke_path():
    """When the plan-execute graph runs end to end, then it executes the step and replans to completion."""
    planner_output = Plan(
        steps=[SingleStep(content="look up account", executor="search")],
        final_response="done",
    )
    replanner_output = Plan(steps=[], final_response="all set")
    named_executor = Mock()
    named_executor.invoke.return_value = {
        "messages": [Mock(content="intermediate"), Mock(content="tool result")],
        "args": {"updated": True},
    }
    impl, _, _ = make_impl(planner_output, replanner_output, executor={"search": named_executor})

    result = impl.graph.invoke(
        {"input": "summarize the plan", "plan": [], "past_steps": [], "response": "", "args": {}}
    )

    assert result["response"] == "all set"
    assert result["past_steps"] == [("look up account", "tool result")]
    assert named_executor.invoke.call_args.kwargs == {"additional_args": {}}
