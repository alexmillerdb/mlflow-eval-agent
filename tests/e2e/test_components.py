"""E2E tests for UI components."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestSidebarComponent:
    def test_runtime_info_displayed(self, app_page: Page):
        """Verify runtime context is shown."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_text("Runtime:")).to_be_visible()

    def test_new_and_clear_buttons(self, app_page: Page):
        """Verify both session control buttons exist."""
        sidebar = app_page.locator('[data-testid="stSidebar"]')
        expect(sidebar.get_by_role("button", name="New")).to_be_visible()
        expect(sidebar.get_by_role("button", name="Clear")).to_be_visible()
