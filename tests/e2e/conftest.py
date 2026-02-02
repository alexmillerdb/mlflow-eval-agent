"""E2E test fixtures for Streamlit app."""
import os
import pytest
import subprocess
import time
import socket
from playwright.sync_api import Page

WEBAPP_TESTING_SCRIPTS = os.path.expanduser(
    "~/.claude/plugins/cache/anthropic-agent-skills/example-skills/00756142ab04/skills/webapp-testing/scripts"
)


def find_free_port():
    """Find an available port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def streamlit_server():
    """Start Streamlit server for E2E tests."""
    port = find_free_port()
    proc = subprocess.Popen([
        "streamlit", "run", "src/app/main.py",
        f"--server.port={port}",
        "--server.headless=true",
        "--server.runOnSave=false",
    ])
    time.sleep(3)  # Wait for server to start
    yield f"http://localhost:{port}"
    proc.terminate()
    proc.wait()


@pytest.fixture
def app_page(page: Page, streamlit_server: str):
    """Navigate to app and wait for load."""
    page.goto(streamlit_server)
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=10000)
    return page


@pytest.fixture
def with_server_script():
    """Path to with_server.py helper."""
    return os.path.join(WEBAPP_TESTING_SCRIPTS, "with_server.py")
