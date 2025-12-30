import cv2
import numpy as np
import streamlit as st
from modules.utils import bgr_to_rgb

def render_ycbcr_display():
    st.header("4. Hiển thị kênh YCbCr")
    # SIDEBAR - UPLOAD
    with st.sidebar:
        st.subheader("Upload ảnh")
        uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'jpeg', 'png'], key="ycbcr")
    if uploaded_file is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Chuyển sang YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        # HÀNG 1: Ảnh RGB gốc ở giữa
        col_left, col_center, col_right = st.columns([1, 2, 1])
        
        with col_center:
            st.markdown("**Ảnh gốc (RGB)**")
            st.image(bgr_to_rgb(image), use_column_width=True)
        
        # HÀNG 2: 3 kênh YCbCr
        col_y, col_cr, col_cb = st.columns(3)
            
        with col_y:
            st.markdown("**Y (Độ sáng)**")
            st.image(y_channel, use_column_width=True)
        with col_cr:
            st.markdown("**Cr (Đỏ)**")
            st.image(cr_channel, use_column_width=True)
        with col_cb:
            st.markdown("**Cb (Xanh)**")
            st.image(cb_channel, use_column_width=True)
