"""E2E tests for chat interface."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestChatFlow:
    def test_page_loads(self, app_page: Page):
        """Verify app renders without errors."""
        expect(app_page.locator("h1")).to_contain_text("MLflow Eval Agent")

    def test_chat_input_visible(self, app_page: Page):
        """Verify chat input is present."""
        chat_input = app_page.locator('[data-testid="stChatInput"]')
        expect(chat_input).to_be_visible()

    def test_submit_message(self, app_page: Page):
        """Verify user message appears in chat."""
        chat_input = app_page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("Hello")
        chat_input.press("Enter")
        # Verify message appears
        user_msg = app_page.locator('[data-testid="stChatMessage"]').first
        expect(user_msg).to_contain_text("Hello")

    def test_streaming_response(self, app_page: Page):
        """Verify streaming text appears incrementally."""
        chat_input = app_page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("List 3 traces")
        chat_input.press("Enter")
        # Wait for assistant response to start
        app_page.wait_for_selector('[data-testid="stChatMessage"]:nth-child(2)', timeout=30000)
        # Response should eventually complete
        expect(app_page.locator('[data-testid="stChatMessage"]')).to_have_count(2, timeout=60000)
