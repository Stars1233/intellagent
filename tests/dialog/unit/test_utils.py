from simulator.dialog.utils import contains_isolated_correct, intermediate_processing


def test_contains_isolated_correct_matches_standalone_token():
    """When CORRECT appears as a standalone token, then the helper returns True."""
    assert contains_isolated_correct("The answer is CORRECT.") is True


def test_contains_isolated_correct_rejects_embedded_token():
    """When CORRECT is embedded inside another word, then the helper returns False."""
    assert contains_isolated_correct("This is INCORRECT.") is False


def test_intermediate_processing_routes_to_chatbot_without_stop_signal():
    """When no stop signal is present, then the dialog continues to the chatbot."""
    state = {"stop_signal": "", "critique_feedback": ""}

    assert intermediate_processing(state) == "chatbot"


def test_intermediate_processing_routes_to_end_critique_without_feedback():
    """When the user has stopped but critique has not responded yet, then the dialog enters critique."""
    state = {"stop_signal": "###STOP", "critique_feedback": ""}

    assert intermediate_processing(state) == "end_critique"


def test_intermediate_processing_routes_to_end_when_feedback_confirms_correctness():
    """When critique feedback contains isolated CORRECT, then the dialog ends."""
    state = {"stop_signal": "###STOP", "critique_feedback": "CORRECT"}

    assert intermediate_processing(state) == "END"


def test_intermediate_processing_routes_back_to_user_when_feedback_is_negative():
    """When critique feedback is negative, then the dialog loops back to the user."""
    state = {"stop_signal": "###STOP", "critique_feedback": "needs another try"}

    assert intermediate_processing(state) == "user"
