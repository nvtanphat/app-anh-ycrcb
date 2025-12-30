"""
Image Processing Experiment Module
Comparing YCrCb vs RGB color spaces
"""
import streamlit as st

st.set_page_config(
    page_title="Image Processing Experiments",
    layout="wide"
)

# Sidebar navigation
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Chon module:",
    ["Skin Detection", "Smart Sharpening", "Noise Removal", "YCbCr Display"]
)

if page == "Skin Detection":
    from modules.skin_detection import render_skin_detection
    render_skin_detection()

elif page == "Smart Sharpening":
    from modules.smart_sharpening import render_smart_sharpening
    render_smart_sharpening()

elif page == "Noise Removal":
    from modules.noise_removal import render_noise_removal
    render_noise_removal()

elif page == "YCbCr Display":
    from modules.ycbcr_display import render_ycbcr_display
    render_ycbcr_display()
