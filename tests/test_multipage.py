import pytest
from unittest.mock import MagicMock, patch

from app_pages.multipage import MultiPage


@pytest.fixture
def mock_streamlit():
    with patch("app_pages.multipage.st") as mock_st:
        # Mock commonly used Streamlit components
        mock_st.sidebar.radio.return_value = "Summary"
        mock_st.query_params = {}

        class SessionStateMock(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError:
                    raise AttributeError(
                        f"'SessionStateMock' has no attribute '{name}'")

            def __setattr__(self, name, value):
                self[name] = value
        mock_st.title = MagicMock()
        mock_st.session_state = SessionStateMock()
        yield mock_st


def test_add_page(mock_streamlit):
    app = MultiPage("Test App")

    def dummy_func(): pass

    app.add_page("Summary", dummy_func)
    assert len(app.pages) == 1
    assert app.pages[0]["title"] == "Summary"
    assert app.pages[0]["function"] == dummy_func


def test_run_calls_correct_function(mock_streamlit):
    app = MultiPage("Test App")

    dummy_called = {"flag": False}

    def dummy_func():
        dummy_called["flag"] = True

    app.add_page("Summary", dummy_func)

    # Simulate session state and sidebar selection
    mock_streamlit.session_state["selected_page"] = "Summary"
    mock_streamlit.sidebar.radio.return_value = "Summary"

    app.run()

    assert dummy_called["flag"] is True


def test_dummy():
    assert 1 == 1
