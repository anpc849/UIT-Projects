import streamlit as st


st.set_page_config(
    page_title="Đồ án môn DS104",
    page_icon="🏠",
)

empty_space = "&nbsp;" * 1500
st.markdown(f"{empty_space}<h3 style='text-align: center;'>Beyond Clicks: Modeling User Behavior to Predict Student Outcomes and Guide Learning</h1>", unsafe_allow_html=True)

st.sidebar.success("Select a demo above.")
