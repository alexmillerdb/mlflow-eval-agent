"""E2E tests for autonomous mode tab."""
import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


class TestAutonomousTab:
    @pytest.fixture(autouse=True)
    def navigate_to_autonomous(self, app_page: Page):
        """Navigate to autonomous tab before each test."""
        app_page.get_by_role("tab", name="Autonomous").click()
        app_page.wait_for_load_state("networkidle")
        yield

    def test_experiment_input_visible(self, app_page: Page):
        """Verify experiment ID input is present."""
        exp_input = app_page.get_by_label("Experiment ID")
        expect(exp_input).to_be_visible()

    def test_max_iterations_input(self, app_page: Page):
        """Verify max iterations control is present."""
        iterations_input = app_page.get_by_label("Max Iterations")
        expect(iterations_input).to_be_visible()
        expect(iterations_input).to_have_value("10")

    def test_start_button_requires_experiment(self, app_page: Page):
        """Verify error shown when no experiment ID."""
        # Clear any existing value
        exp_input = app_page.get_by_label("Experiment ID").first
        exp_input.clear()

        # Click start
        app_page.get_by_role("button", name="Start Autonomous Run").click()

        # Should show error
        expect(app_page.get_by_text("Please enter an Experiment ID")).to_be_visible()

    def test_progress_section_visible(self, app_page: Page):
        """Verify progress section is present."""
        # Progress info should be visible
        expect(app_page.get_by_text("No tasks yet")).to_be_visible()

    def test_task_details_expander(self, app_page: Page):
        """Verify task details expander works."""
        expander = app_page.get_by_text("Task Details")
        expect(expander).to_be_visible()
