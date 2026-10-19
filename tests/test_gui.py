"""
tests/test_gui.py
=================
Tests for the Streamlit GUI using streamlit.testing.v1.AppTest.
"""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_app_loads_smoke():
    app_path = ROOT / "app" / "app.py"
    assert app_path.exists(), f"{app_path} does not exist"

    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()
    assert not at.exception, f"App raised exception: {at.exception}"

    # Verify header rendered
    assert len(at.markdown) > 0


def test_streamlit_app_load_demo():
    app_path = ROOT / "app" / "app.py"
    at = AppTest.from_file(str(app_path), default_timeout=30)
    at.run()

    # Navigate to page 2 (Data Upload & Validation)
    at.sidebar.radio[0].set_value("2. Data Upload & Validation").run()
    assert not at.exception

    # Find the Load Demo button and click it
    demo_btn = at.button(key=None)
    assert len(at.button) > 0
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["demo_loaded"] is True
