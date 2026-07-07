from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from simulator.agents_graphs.dialog_graph import Dialog, set_user_message


def make_dialog(user_response, chatbot_response, critique_response, memory=None, intermediate_processing=None):
    user = Mock()
    user.invoke.return_value = user_response
    chatbot = Mock()
    chatbot.invoke.return_value = chatbot_response
    critique = Mock()
    critique.invoke.return_value = critique_response
    return Dialog(
        user=user,
        chatbot=chatbot,
        critique=critique,
        intermediate_processing=intermediate_processing or (lambda state: END),
        memory=memory,
    )


def test_user_end_condition_routes_to_critique_on_stop_signal():
    """When the user stop signal is present, then the dialog routes to critique instead of the chatbot."""
    dialog = make_dialog({}, {}, {})
    assert dialog.user_end_condition({"stop_signal": "###STOP"}) == "end_critique"
    assert dialog.user_end_condition({"stop_signal": ""}) == "chatbot"


def test_critique_end_condition_uses_intermediate_processing():
    """When intermediate processing ends the dialog, then critique routes to END; otherwise it loops back to user."""
    dialog = make_dialog({}, {}, {}, intermediate_processing=lambda state: "END")
    assert dialog.critique_end_condition({"stop_signal": "###STOP"}) == END

    dialog = make_dialog({}, {}, {}, intermediate_processing=lambda state: "user")
    assert dialog.critique_end_condition({"stop_signal": "###STOP"}) == "user"


def test_set_user_message_without_feedback():
    """When there is no critique feedback, then the user prompt contains only the conversation summary."""
    messages = set_user_message(
        {
            "chatbot_messages": [HumanMessage(content="hello"), AIMessage(content="world")],
            "critique_feedback": "",
            "user_thoughts": [],
            "stop_signal": "",
        }
    )

    assert len(messages) == 1
    assert messages[0].type == "human"
    assert "Conversation" in messages[0].content


def test_set_user_message_with_feedback_adds_retry_prompt():
    """When critique feedback exists, then the retry prompt includes the prior response and feedback."""
    messages = set_user_message(
        {
            "chatbot_messages": [HumanMessage(content="hello"), AIMessage(content="world")],
            "critique_feedback": "Please stop earlier",
            "user_thoughts": ["Thought: because"],
            "stop_signal": "###STOP",
        }
    )

    assert len(messages) == 3
    assert messages[1].type == "ai"
    assert "User Response" in messages[1].content
    assert messages[2].type == "human"
    assert "Please stop earlier" in messages[2].content


def test_simulated_user_node_records_stop_signal_and_thoughts():
    """When the user model returns a stop signal, then the node stores thoughts and stop state in memory."""
    memory = SimpleNamespace(thoughts=[], dialog=[], tools=[])
    memory.insert_thought = Mock(side_effect=lambda thread_id, thought: memory.thoughts.append((thread_id, thought)))
    memory.insert_dialog = Mock(side_effect=lambda thread_id, role, content: memory.dialog.append((thread_id, role, content)))
    memory.insert_tool = Mock(side_effect=lambda thread_id, name, args, output: memory.tools.append((thread_id, name, args, output)))

    dialog = make_dialog(
        user_response={"response": "###STOP", "thought": "I should stop"},
        chatbot_response={},
        critique_response={},
        memory=memory,
    )

    result = dialog.simulated_user_node(
        {
            "user_messages": [HumanMessage(content="start")],
            "chatbot_messages": [HumanMessage(content="hello")],
            "user_thoughts": [],
            "thread_id": "thread-1",
            "critique_feedback": "",
            "stop_signal": "",
        }
    )

    assert result["stop_signal"] == "###STOP"
    assert result["user_thoughts"] == ["I should stop"]
    assert memory.thoughts == [("thread-1", "I should stop")]
    assert memory.dialog == [("thread-1", "Human", "###STOP")]


def test_chat_bot_node_returns_last_ai_message_and_records_messages():
    """When the chatbot emits tool and final messages, then the node returns the post-human slice and final answer."""
    chatbot_response = {
        "messages": [
            HumanMessage(content="start"),
            AIMessage(content="tool output"),
            AIMessage(content="final answer"),
        ]
    }
    dialog = make_dialog(user_response={}, chatbot_response=chatbot_response, critique_response={})

    result = dialog.chat_bot_node(
        {
            "chatbot_messages": [HumanMessage(content="start")],
            "chatbot_args": {"x": 1},
            "thread_id": "thread-1",
        }
    )

    assert len(result["chatbot_messages"]) == 2
    assert result["chatbot_messages"][-1].content == "final answer"
    assert result["user_messages"][0].content == "final answer"


def test_critique_node_uses_last_user_thought():
    """When critique runs, then the last user thought is passed into the critique reasoning payload."""
    critique = Mock()
    critique.invoke.return_value = SimpleNamespace(content="CORRECT")
    dialog = make_dialog(user_response={}, chatbot_response={}, critique_response={}, intermediate_processing=lambda state: END)
    dialog.critique = critique

    result = dialog.critique_node(
        {
            "user_thoughts": ["Thought: they followed the policy"],
            "chatbot_messages": [HumanMessage(content="hi"), AIMessage(content="done")],
        }
    )

    assert result == {"critique_feedback": "CORRECT"}
    assert "they followed the policy" in critique.invoke.call_args.args[0]["reason"]
