from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_visual_diagnosis_assistant import page_visual_diagnosis_assistant_body
from app_pages.page_summary import page_summary_body

app = MultiPage(app_name="Mildew Detector")  # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Diagnosis Assistant", page_visual_diagnosis_assistant_body)
app.add_page("Summary", page_summary_body)

if __name__ == "__main__":
    app.run()
