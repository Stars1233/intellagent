from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

from langchain_core.messages import HumanMessage

import simulator.dialog.dialog_manager as dialog_manager_module
from simulator.dialog.dialog_manager import DialogManager


class DummyCallback:
    def __enter__(self):
        self.total_cost = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_dialog_manager_smoke_run_event_persists_conversation(monkeypatch, tmp_path):
    """When the real dialog graph runs with stubbed LLMs, then sqlite persistence records the conversation."""
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
        "num_workers": 1,
        "timeout": 5,
    }

    user_base_llm = MagicMock(name="user_base_llm")
    user_chain = MagicMock(name="user_chain")
    user_chain.invoke.return_value = {"response": "###STOP", "thought": "I should stop"}
    user_base_llm.__or__.return_value = user_chain

    critique_base_llm = MagicMock(name="critique_base_llm")
    critique_chain = MagicMock(name="critique_chain")
    critique_chain.invoke.return_value = SimpleNamespace(content="CORRECT")
    critique_prompt = MagicMock(name="critique_prompt")
    critique_prompt.partial.return_value = critique_prompt
    critique_prompt.__or__.return_value = critique_chain

    chat_base_llm = MagicMock(name="chat_base_llm")
    chatbot_stub = MagicMock(name="chatbot_stub")
    chatbot_prompt = MagicMock(name="chatbot_prompt")
    chatbot_prompt.format_messages.return_value = [HumanMessage(content="system prompt")]
    user_prompt = MagicMock(name="user_prompt")
    user_prompt.format_messages.return_value = [HumanMessage(content="user prompt")]

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
            return critique_prompt
        if template == "system prompt":
            return chatbot_prompt
        if template == "user prompt":
            return user_prompt
        raise AssertionError(f"unexpected prompt args: {args}")

    monkeypatch.setattr(dialog_manager_module, "get_llm", fake_get_llm)
    monkeypatch.setattr(dialog_manager_module, "get_prompt_template", fake_get_prompt_template)
    monkeypatch.setattr(dialog_manager_module, "set_callback", lambda _type: DummyCallback)
    monkeypatch.setattr(dialog_manager_module, "AgentTools", MagicMock(return_value=chatbot_stub))
    monkeypatch.setattr(dialog_manager_module.uuid, "uuid4", lambda: UUID("12345678-1234-5678-1234-567812345678"))

    manager = DialogManager(config=config, environment=env)
    manager.init_dialog(str(tmp_path))

    event = SimpleNamespace(
        scenario="user asks to stop",
        relevant_rows=["row 1"],
        description=SimpleNamespace(expected_behaviour="stop early"),
        database={"users": []},
        id=1,
    )

    result = manager.run_event(event)
    thread_id = str(UUID("12345678-1234-5678-1234-567812345678"))

    assert result["stop_signal"] == "###STOP"
    assert result["critique_feedback"] == "CORRECT"
    assert manager.memory.read_thought(thread_id) == [(thread_id, "I should stop")]
    assert manager.memory.read_dialog(thread_id) == [(thread_id, "Human", "###STOP")]
    manager.memory.exit()
