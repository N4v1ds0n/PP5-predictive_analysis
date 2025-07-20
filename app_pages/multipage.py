import streamlit as st


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        """
        Initializes the multi-page application.
        """
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🍒",
            layout="wide",)

    def add_page(self, title, func) -> None:
        """
        Adds a new page to the multi-page app.

        Parameters:
        - title (str): The name of the page.
        - func (function): The function that renders the page.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        # Retrieve query parameters using st.query_params
        query_params = st.query_params
        page_param = query_params.get("page", None)

        # Find page by title match
        page_titles = [p["title"] for p in self.pages]
        page = next(
            (p for p in self.pages if p["title"].lower() == page_param), 
            self.pages[0]
        )

        # Sidebar with synced selection
        selected_title = st.sidebar.radio(
            "📂 Menu",
            page_titles,
            index=page_titles.index(page["title"])
        )

        # Update query params (URL) when user changes selection
        if selected_title.lower() != page_param:
            st.query_params["page"] = selected_title.lower()

        # App title
        st.title(self.app_name)

        # Render selected page
        for p in self.pages:
            if p["title"] == selected_title:
                p["function"]()