"""
Tests for Gradio chat interface.

These tests verify that students correctly implement the Gradio UI components
and integrate the chat functionality properly. Tests run without launching
the UI server by inspecting the component tree and mocking dependencies.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import gradio as gr

from multimodal_moderation.gradio_app import (
    PERSONA_INSTRUCTIONS,
    _analytics_rows,
    _record_moderation_result,
    check_content_safety,
    create_chat_interface,
    ChatSessionWithTracing,
)


def test_create_chat_interface_returns_blocks():
    """Verify that create_chat_interface returns a Gradio Blocks object"""
    demo = create_chat_interface()

    assert isinstance(demo, gr.Blocks), f"create_chat_interface should return gr.Blocks, got {type(demo)}"


def test_chatbot_uses_messages_type():
    """Verify that Chatbot is configured to use 'messages' type"""
    demo = create_chat_interface()

    chatbot = None
    for block in demo.blocks.values():
        if isinstance(block, gr.Chatbot):
            chatbot = block
            break

    assert chatbot is not None, "Chatbot component not found"
    assert chatbot.type == "messages", "Chatbot should use type='messages' for multimodal support"


def test_chatbot_has_copy_button():
    """Verify that Chatbot has copy button enabled"""
    demo = create_chat_interface()

    chatbot = None
    for block in demo.blocks.values():
        if isinstance(block, gr.Chatbot):
            chatbot = block
            break

    assert chatbot is not None, "Chatbot component not found"
    assert chatbot.show_copy_button == True, "Chatbot should have show_copy_button=True"


def test_demo_has_chatbot_component():
    """Verify that the demo contains a Chatbot component"""
    demo = create_chat_interface()

    chatbots = [block for block in demo.blocks.values() if isinstance(block, gr.Chatbot)]

    assert len(chatbots) > 0, "Demo should contain at least one gr.Chatbot component"


def test_demo_has_multimodal_textbox():
    """Verify that the demo uses MultimodalTextbox for input"""
    demo = create_chat_interface()

    textboxes = [block for block in demo.blocks.values() if isinstance(block, gr.MultimodalTextbox)]

    assert len(textboxes) > 0, "Demo should contain a gr.MultimodalTextbox for file uploads"


def test_demo_has_persona_selector_and_analytics_dashboard():
    demo = create_chat_interface()
    dropdowns = [
        block for block in demo.blocks.values() if isinstance(block, gr.Dropdown) and block.label == "Customer persona"
    ]
    dashboards = [
        block
        for block in demo.blocks.values()
        if isinstance(block, gr.Dataframe) and block.label == "Aggregate moderation counts"
    ]

    assert len(dropdowns) == 1
    assert dropdowns[0].value == "Angry customer (default)"
    assert {choice[0] for choice in dropdowns[0].choices} == set(PERSONA_INSTRUCTIONS)
    assert len(dashboards) == 1


def test_analytics_counts_flags_without_storing_content():
    analytics = {}
    _record_moderation_result(
        analytics,
        {
            "rationale": "Contains personal data",
            "contains_pii": True,
            "is_unfriendly": False,
            "transcription": "Sensitive audio content",
        },
    )

    rows = _analytics_rows(analytics)
    assert ["Total checks", 1] in rows
    assert ["Flagged checks", 1] in rows
    assert ["Contains Pii", 1] in rows
    assert "rationale" not in analytics
    assert "transcription" not in analytics


def test_misinformation_is_reviewed_but_not_blocked():
    result = {
        "rationale": "The claim cannot be verified.",
        "contains_pii": False,
        "is_unfriendly": False,
        "is_unprofessional": False,
        "contains_hate_speech": False,
        "is_spam": False,
        "contains_misinformation": True,
    }
    analytics = {}

    with patch("multimodal_moderation.gradio_app._call_text_moderation") as mock_moderate:
        mock_moderate.return_value = (result, result["rationale"], "text", "text/plain")
        with patch("multimodal_moderation.gradio_app.tracer.start_as_current_span") as span:
            span.return_value.__enter__.return_value = MagicMock()
            is_safe, feedback, _ = check_content_safety(text="A factual claim.", analytics=analytics)

    assert is_safe
    assert "review recommended" in feedback.lower()
    assert analytics["contains_misinformation"] == 1


def test_multimodal_textbox_accepts_multiple_files():
    """Verify that MultimodalTextbox is configured to accept multiple files"""
    demo = create_chat_interface()

    textbox = None
    for block in demo.blocks.values():
        if isinstance(block, gr.MultimodalTextbox):
            textbox = block
            break

    assert textbox is not None, "MultimodalTextbox component not found"
    assert textbox.file_count == "multiple", "MultimodalTextbox should allow multiple files (file_count='multiple')"


def test_multimodal_textbox_accepts_correct_file_types():
    """Verify that MultimodalTextbox accepts image, video, and audio files"""
    demo = create_chat_interface()

    textbox = None
    for block in demo.blocks.values():
        if isinstance(block, gr.MultimodalTextbox):
            textbox = block
            break

    assert textbox is not None, "MultimodalTextbox component not found"

    expected_types = ["image", "video", "audio"]
    for file_type in expected_types:
        assert file_type in textbox.file_types, f"MultimodalTextbox should accept '{file_type}' files"


async def test_chat_with_gemini_calls_moderation():
    """Verify that chat_with_gemini integrates with moderation service"""
    # Mock the moderation API
    with patch("multimodal_moderation.gradio_app.check_content_safety") as mock_moderation:
        # Mock successful moderation
        mock_moderation.return_value = (True, "Content passed moderation", "text/plain")

        # Mock the agent run
        with patch("multimodal_moderation.gradio_app.customer_agent.run") as mock_agent:
            mock_result = MagicMock()
            mock_result.output = "AI response"
            mock_result.all_messages.return_value = []
            mock_agent.return_value = mock_result

            message = {"text": "Hello"}
            history = []
            past_messages = []

            chat_session = ChatSessionWithTracing()
            result, messages, feedback, dashboard, analytics = await chat_session.chat_with_gemini(
                message, history, past_messages, persona="Calm but dissatisfied"
            )

            # Verify moderation was called
            assert mock_moderation.called, "chat_with_gemini should call check_content_safety"

            # Verify it was called with correct arguments
            mock_moderation.assert_called_with(text="Hello", analytics={})
            mock_agent.assert_awaited_once()
            assert PERSONA_INSTRUCTIONS["Calm but dissatisfied"] in mock_agent.await_args.args[0][0]
            assert ["Total checks", 0] in dashboard
            assert analytics == {}


async def test_chat_with_gemini_blocks_flagged_content():
    """Verify that flagged content is blocked and not sent to AI"""
    with patch("multimodal_moderation.gradio_app.check_content_safety") as mock_moderation:
        # Mock failed moderation
        mock_moderation.return_value = (False, "Content flagged: unfriendly", "text/plain")

        with patch("multimodal_moderation.gradio_app.customer_agent.run") as mock_agent:
            message = {"text": "Inappropriate message"}
            history = []
            past_messages = []

            chat_session = ChatSessionWithTracing()
            result, messages, feedback, dashboard, analytics = await chat_session.chat_with_gemini(
                message, history, past_messages
            )

            # Verify agent was NOT called (content blocked)
            assert not mock_agent.called, "Agent should not be called when content is flagged"

            # Verify feedback contains warning
            assert "flagged" in feedback.lower(), "Feedback should indicate content was flagged"


@pytest.mark.parametrize("flag", ["contains_hate_speech", "is_spam"])
def test_hate_speech_and_spam_are_blocking_flags(flag):
    result = {
        "rationale": "Flagged by moderation.",
        "contains_pii": False,
        "is_unfriendly": False,
        "is_unprofessional": False,
        "contains_hate_speech": False,
        "is_spam": False,
        "contains_misinformation": False,
    }
    result[flag] = True

    with patch("multimodal_moderation.gradio_app._call_text_moderation") as mock_moderate:
        mock_moderate.return_value = (result, result["rationale"], "text", "text/plain")
        with patch("multimodal_moderation.gradio_app.tracer.start_as_current_span") as span:
            span.return_value.__enter__.return_value = MagicMock()
            is_safe, feedback, _ = check_content_safety(text="A message.")

    assert not is_safe
    assert "flagged" in feedback.lower()
