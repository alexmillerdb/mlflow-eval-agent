"""E2E tests for sidebar configuration."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestSidebar:
    def test_sidebar_renders(self, app_page: Page):
        """Verify sidebar is visible."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible()

    def test_experiment_id_input(self, app_page: Page):
        """Verify experiment ID can be entered."""
        exp_input = app_page.locator('[data-testid="stSidebar"] input').first
        exp_input.fill("12345")
        expect(exp_input).to_have_value("12345")

    def test_new_session_button(self, app_page: Page):
        """Verify new session button clears chat."""
        # Add a message first
        chat_input = app_page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("Test message")
        chat_input.press("Enter")
        app_page.wait_for_selector('[data-testid="stChatMessage"]')
        # Click new session
        app_page.locator('button:has-text("New")').click()
        # Chat should be cleared
        expect(app_page.locator('[data-testid="stChatMessage"]')).to_have_count(0, timeout=5000)
