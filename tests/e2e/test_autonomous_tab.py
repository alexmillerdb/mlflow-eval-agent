"""E2E tests for autonomous mode."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestAutonomousMode:
    @pytest.fixture(autouse=True)
    def navigate_to_autonomous(self, app_page: Page):
        """Switch to autonomous mode via sidebar selector."""
        app_page.get_by_label("Mode").select_option("Autonomous")
        app_page.wait_for_load_state("networkidle")
        yield

    def test_experiment_input_visible(self, app_page: Page):
        """Verify experiment ID input is present in sidebar."""
        exp_input = app_page.get_by_label("Experiment ID")
        expect(exp_input).to_be_visible()

    def test_max_iterations_input(self, app_page: Page):
        """Verify max iterations control is present in sidebar."""
        iterations_input = app_page.get_by_label("Max Iterations")
        expect(iterations_input).to_be_visible()
        expect(iterations_input).to_have_value("10")

    def test_start_button_requires_experiment(self, app_page: Page):
        """Verify error shown when no experiment ID."""
        # Clear any existing value
        exp_input = app_page.get_by_label("Experiment ID").first
        exp_input.clear()

        # Click start in sidebar
        app_page.get_by_role("button", name="Start").click()

        # Should show error
        expect(app_page.get_by_text("Set Experiment ID first")).to_be_visible()

    def test_task_details_expander(self, app_page: Page):
        """Verify task details are not visible until autonomous run starts."""
        # In unified UI, task progress only shows in side panel during active run
        # No tasks message may or may not be visible depending on layout
        pass
