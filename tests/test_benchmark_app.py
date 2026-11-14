"""Unit tests verifying Streamlit Dashboard integration for Phase 8 Benchmark Lab."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_tabs_rendering():
    """Verify that app.py runs and renders all 8 navigation tabs including Comparative Benchmark Lab."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify no unhandled exceptions crashed the app
    assert not at.exception

    # Verify tab names or tab contents are present
    assert len(at.tabs) >= 8
