"""E2E tests for mode toggle (unified UI)."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestModeToggle:
    def test_mode_selector_visible(self, app_page: Page):
        """Verify mode selectbox is present in sidebar."""
        mode_select = app_page.get_by_label("Mode")
        expect(mode_select).to_be_visible()

    def test_switch_to_autonomous_mode(self, app_page: Page):
        """Verify switching to autonomous shows controls in sidebar."""
        app_page.get_by_label("Mode").select_option("Autonomous")
        app_page.wait_for_load_state("networkidle")

        # Autonomous controls should appear in sidebar
        expect(app_page.get_by_label("Max Iterations")).to_be_visible()
        expect(app_page.get_by_role("button", name="Start")).to_be_visible()

    def test_switch_to_interactive_mode(self, app_page: Page):
        """Verify switching back to interactive hides autonomous controls."""
        # Switch to autonomous first
        app_page.get_by_label("Mode").select_option("Autonomous")
        app_page.wait_for_load_state("networkidle")

        # Switch back
        app_page.get_by_label("Mode").select_option("Interactive")
        app_page.wait_for_load_state("networkidle")

        # Chat input should be visible, no autonomous controls
        expect(app_page.locator('[data-testid="stChatInput"]')).to_be_visible()
        expect(app_page.get_by_role("button", name="Start")).not_to_be_visible()
