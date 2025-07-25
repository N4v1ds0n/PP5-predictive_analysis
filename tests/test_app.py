from app_pages.multipage import MultiPage
from app_pages.page_visual_diagnosis_assistant import (
    page_visual_diagnosis_assistant_body)
from app_pages.page_summary import page_summary_body
from app_pages.page_mildew_detector import page_mildew_detector_body
from app_pages.page_model_performance import page_ml_performance_body
from app_pages.page_working_hypothesis import page_working_hypothesis_body


def test_app_page_registration():
    app = MultiPage(app_name="Mildew Detector")

    app.add_page("Summary", page_summary_body)
    app.add_page("Diagnosis Assistant", page_visual_diagnosis_assistant_body)
    app.add_page("Mildew Detector", page_mildew_detector_body)
    app.add_page("Model Performance Metrics", page_ml_performance_body)
    app.add_page("Working Hypotheses", page_working_hypothesis_body)

    assert len(app.pages) == 5
    assert app.pages[0]['title'] == "Summary"
    assert app.pages[1]['function'] == page_visual_diagnosis_assistant_body
