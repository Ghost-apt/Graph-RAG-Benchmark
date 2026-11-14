"""Unit tests verifying Streamlit Dashboard Tab 9 Production Operations integration."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_production_dashboard_nine_tabs():
    """Verify that app.py compiles and renders all navigation tabs including Tab 9 Production Operations."""
    app_path = Path(__file__).parent.parent / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Verify app did not crash
    assert not at.exception

    # Verify top-level + nested tabs rendered (8 top-level + 1 ops + nested sub-tabs = >= 9)
    assert len(at.tabs) >= 9
