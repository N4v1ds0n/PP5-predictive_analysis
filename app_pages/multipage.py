import streamlit as st


class MultiPage:
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🍒",
            layout="wide",
        )

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        page_titles = [p["title"] for p in self.pages]
        query_params = st.query_params

        # Normalize title for URL matching
        def normalize(text):
            return text.strip().lower().replace(" ", "-")

        # --- Initialize session state from query param (only once)
        if "selected_page" not in st.session_state:
            if "page" in query_params:
                matched = [
                    title for title in page_titles
                    if normalize(title) == query_params["page"]
                ]
                st.session_state.selected_page = matched[0] if matched else page_titles[0]
            else:
                st.session_state.selected_page = page_titles[0]

        # --- Sidebar
        selected_title = st.sidebar.radio(
            "📂 Menu",
            options=page_titles,
            index=page_titles.index(st.session_state.selected_page),
            key="selected_page",
        )

        # --- Sync session state back to query params
        st.query_params["page"] = normalize(selected_title)

        # --- App title
        st.title(self.app_name)

        # --- Run the selected page function
        for p in self.pages:
            if p["title"] == selected_title:
                p["function"]()
