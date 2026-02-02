"""E2E tests for tab navigation."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestTabs:
    def test_tabs_visible(self, app_page: Page):
        """Verify both tabs are present."""
        interactive_tab = app_page.get_by_role("tab", name="Interactive")
        autonomous_tab = app_page.get_by_role("tab", name="Autonomous")
        expect(interactive_tab).to_be_visible()
        expect(autonomous_tab).to_be_visible()

    def test_switch_to_autonomous_tab(self, app_page: Page):
        """Verify switching to autonomous tab shows controls."""
        app_page.get_by_role("tab", name="Autonomous").click()
        app_page.wait_for_load_state("networkidle")

        # Should see autonomous controls
        expect(app_page.get_by_text("Autonomous Evaluation")).to_be_visible()
        expect(app_page.get_by_role("button", name="Start Autonomous Run")).to_be_visible()

    def test_switch_back_to_interactive(self, app_page: Page):
        """Verify switching back to interactive shows chat."""
        app_page.get_by_role("tab", name="Autonomous").click()
        app_page.wait_for_load_state("networkidle")

        app_page.get_by_role("tab", name="Interactive").click()
        app_page.wait_for_load_state("networkidle")

        expect(app_page.locator('[data-testid="stChatInput"]')).to_be_visible()
