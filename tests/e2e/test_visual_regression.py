"""Visual regression tests for UI consistency."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = [pytest.mark.e2e, pytest.mark.visual]


class TestVisualRegression:
    def test_main_page_appearance(self, app_page: Page):
        """Capture main page baseline."""
        app_page.wait_for_load_state("networkidle")
        expect(app_page).to_have_screenshot("main_page.png", full_page=True)

    def test_sidebar_appearance(self, app_page: Page):
        """Capture sidebar baseline."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_have_screenshot("sidebar.png")

    def test_chat_with_message(self, app_page: Page):
        """Capture chat with user message."""
        chat_input = app_page.locator('[data-testid="stChatInput"] textarea')
        chat_input.fill("Hello")
        chat_input.press("Enter")
        app_page.wait_for_selector('[data-testid="stChatMessage"]')
        expect(app_page).to_have_screenshot("chat_with_message.png", full_page=True)
