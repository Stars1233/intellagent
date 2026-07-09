from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
import pytest

import simulator.dialog.dialog_manager as dialog_manager_module
from simulator.dialog.dialog_manager import DialogManager


class DummyCallback:
    def __enter__(self):
        self.total_cost = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def build_manager(monkeypatch, chatbot_messages_result=None, user_messages_result=None):
    env = SimpleNamespace(
        data_examples={"users": '{"id": 1}'},
        data_schema={"users": ["id"]},
        tools=[],
        tools_schema=None,
        prompt="system prompt",
    )
    config = {
        "llm_user": {"type": "openai", "name": "user-model"},
        "llm_chat": {"type": "openai", "name": "chat-model"},
        "user_parsing_mode": "thought",
        "critique_config": {"llm": {"type": "openai", "name": "critique-model"}, "prompt": {"from_str": {"template": "critique prompt"}}},
        "user_prompt": {"from_str": {"template": "user prompt"}},
        "recursion_limit": 7,
        "num_workers": 2,
        "timeout": 11,
    }

    user_base_llm = MagicMock(name="user_base_llm")
    user_chain = MagicMock(name="user_chain")
    user_base_llm.__or__.return_value = user_chain

    critique_base_llm = MagicMock(name="critique_base_llm")
    critique_chain = MagicMock(name="critique_chain")
    critique_chain.invoke.return_value = SimpleNamespace(content="CORRECT")

    chat_base_llm = MagicMock(name="chat_base_llm")
    chatbot_stub = MagicMock(name="chatbot_stub")
    memory_stub = MagicMock(name="memory_stub")
    dialog_stub = MagicMock(name="dialog_stub")

    if chatbot_messages_result is None:
        chatbot_messages_result = [HumanMessage(content="system prompt")]
    if user_messages_result is None:
        user_messages_result = [HumanMessage(content="user prompt")]

    def fake_get_llm(llm_config):
        if llm_config["name"] == "user-model":
            return user_base_llm
        if llm_config["name"] == "critique-model":
            return critique_base_llm
        if llm_config["name"] == "chat-model":
            return chat_base_llm
        raise AssertionError(f"unexpected llm config: {llm_config}")

    def fake_get_prompt_template(args):
        template = args.get("from_str", {}).get("template")
        if template == "critique prompt":
            prompt = MagicMock(name="critique_prompt")
            prompt.partial.return_value = prompt
            prompt.__or__.return_value = critique_chain
            return prompt
        if template == "system prompt":
            prompt = MagicMock(name="chatbot_prompt")
            prompt.format_messages.return_value = chatbot_messages_result
            return prompt
        if template == "user prompt":
            prompt = MagicMock(name="user_prompt")
            prompt.format_messages.return_value = user_messages_result
            return prompt
        raise AssertionError(f"unexpected prompt args: {args}")

    monkeypatch.setattr(dialog_manager_module, "get_llm", fake_get_llm)
    monkeypatch.setattr(dialog_manager_module, "get_prompt_template", fake_get_prompt_template)
    monkeypatch.setattr(dialog_manager_module, "set_callback", lambda _type: DummyCallback)
    monkeypatch.setattr(dialog_manager_module, "AgentTools", MagicMock(return_value=chatbot_stub))
    monkeypatch.setattr(dialog_manager_module, "SqliteSaver", MagicMock(return_value=memory_stub))
    monkeypatch.setattr(dialog_manager_module, "Dialog", MagicMock(return_value=dialog_stub))

    manager = DialogManager(config=config, environment=env)
    return manager, {
        "config": config,
        "env": env,
        "user_base_llm": user_base_llm,
        "user_chain": user_chain,
        "critique_base_llm": critique_base_llm,
        "critique_chain": critique_chain,
        "chat_base_llm": chat_base_llm,
        "chatbot_stub": chatbot_stub,
        "memory_stub": memory_stub,
        "dialog_stub": dialog_stub,
    }


def test_get_user_parsing_function_default_returns_raw_message(monkeypatch):
    """When parsing is in default mode, then the parser returns the raw content and an empty thought."""
    manager, _ = build_manager(monkeypatch)

    result = manager.get_user_parsing_function("default")(AIMessage(content="plain response"))

    assert result == {"response": "plain response", "thought": ""}


def test_get_user_parsing_function_thought_mode_splits_thought_and_response(monkeypatch):
    """When parsing is in thought mode, then the parser separates the thought from the user response."""
    manager, _ = build_manager(monkeypatch)

    result = manager.get_user_parsing_function("thought")(
        AIMessage(content="I want to stop here\nUser Response: ###STOP")
    )

    assert result == {"response": "###STOP", "thought": "I want to stop here"}


def test_get_user_parsing_function_thought_mode_falls_back_when_pattern_is_missing(monkeypatch):
    """When parsing is in thought mode but the delimiter is missing, then the parser falls back safely."""
    manager, _ = build_manager(monkeypatch)

    result = manager.get_user_parsing_function("thought")(AIMessage(content="plain response"))

    assert result == {"response": "plain response", "thought": ""}


def test_set_agent_tool_chatbot_appends_greeting_for_single_message_prompt(monkeypatch):
    """When the chatbot prompt renders a single message, then the manager appends the default greeting."""
    manager, handles = build_manager(monkeypatch, chatbot_messages_result=[HumanMessage(content="system prompt")])
    manager.chatbot_initial_messages = None

    manager.set_agent_tool_chatbot({"topic": "billing"})

    assert manager.chatbot is handles["chatbot_stub"]
    assert len(manager.chatbot_initial_messages) == 2
    assert manager.chatbot_initial_messages[0].content == "system prompt"
    assert manager.chatbot_initial_messages[1].content == "Hello! 👋 I'm here to help with any request you might have."


def test_set_agent_tool_chatbot_preserves_multi_message_prompt(monkeypatch):
    """When the chatbot prompt already contains multiple messages, then no greeting is appended."""
    manager, handles = build_manager(
        monkeypatch,
        chatbot_messages_result=[HumanMessage(content="system"), AIMessage(content="assistant intro")],
    )
    manager.chatbot_initial_messages = None

    manager.set_agent_tool_chatbot({"topic": "billing"})

    assert manager.chatbot is handles["chatbot_stub"]
    assert len(manager.chatbot_initial_messages) == 2
    assert manager.chatbot_initial_messages[-1].content == "assistant intro"


def test_init_dialog_wires_memory_dialog_and_user_prompt(monkeypatch, tmp_path):
    """When dialog initialization runs, then memory, dialog wiring, and the user prompt are all configured."""
    manager, handles = build_manager(monkeypatch)

    manager.init_dialog(str(tmp_path))

    dialog_kwargs = dialog_manager_module.Dialog.call_args.kwargs
    assert manager.memory is handles["memory_stub"]
    assert dialog_kwargs["memory"] is handles["memory_stub"]
    assert dialog_kwargs["intermediate_processing"].__name__ == "intermediate_processing"
    assert manager.dialog is handles["dialog_stub"]
    assert manager.user_prompt is not None


def test_run_raises_when_dialog_is_not_initialized(monkeypatch):
    """When run is called before init_dialog, then the manager raises a clear error."""
    manager, _ = build_manager(monkeypatch)

    manager.dialog = None

    with pytest.raises(ValueError, match="dialog is not initialized"):
        manager.run()


def test_run_forwards_formatted_messages_and_recursion_limit(monkeypatch):
    """When run executes, then it forwards prompt messages, chatbot state, and recursion settings to the dialog."""
    manager, _ = build_manager(monkeypatch, user_messages_result=[HumanMessage(content="user prompt")])
    dialog_stub = MagicMock()
    manager.dialog = dialog_stub
    manager.chatbot_initial_messages = [HumanMessage(content="system prompt")]
    manager.user_prompt = MagicMock()
    manager.user_prompt.format_messages.return_value = [HumanMessage(content="user prompt")]

    result = manager.run(user_prompt_params={"scenario": "demo"}, chatbot_env_args={"data": {"users": []}})

    assert result is dialog_stub.invoke.return_value
    payload = dialog_stub.invoke.call_args.kwargs["input"]
    assert payload["user_messages"][0].content == "user prompt"
    assert payload["chatbot_messages"][0].content == "system prompt"
    assert payload["chatbot_args"] == {"data": {"users": []}}
    assert payload["user_thoughts"] == []
    assert isinstance(payload["thread_id"], str)
    assert dialog_stub.invoke.call_args.kwargs["config"] == {"recursion_limit": 7}


def test_run_event_maps_event_fields_into_run_arguments(monkeypatch):
    """When an event is executed, then its scenario, rows, and expected behavior are mapped into run parameters."""
    manager, _ = build_manager(monkeypatch)
    manager.run = MagicMock(return_value={"status": "ok"})
    event = SimpleNamespace(
        scenario="scenario text",
        relevant_rows=["row 1"],
        description=SimpleNamespace(expected_behaviour="expected behavior"),
        database={"users": []},
    )

    result = manager.run_event(event)

    assert result == {"status": "ok"}
    manager.run.assert_called_once_with(
        user_prompt_params={
            "scenario": "scenario text",
            "rows": ["row 1"],
            "expected_behaviour": "expected behavior",
        },
        chatbot_env_args={"data": {"users": []}},
    )


def test_run_events_aggregates_results_and_cost(monkeypatch):
    """When batch execution returns mixed results, then successful runs are preserved and costs are summed."""
    manager, _ = build_manager(monkeypatch)
    events = [SimpleNamespace(id=11), SimpleNamespace(id=22)]
    monkeypatch.setattr(
        dialog_manager_module,
        "async_batch_invoke",
        lambda *args, **kwargs: [
            {"index": 0, "result": {"ok": True}, "usage": 3, "error": None},
            {"index": 1, "result": None, "usage": 0, "error": "Timeout"},
        ],
    )

    result, cost = manager.run_events(events)

    assert result == [{"res": {"ok": True}, "event_id": 11}]
    assert cost == 3
