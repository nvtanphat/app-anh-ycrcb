# Import thư viện xử lý ảnh OpenCV
import cv2
# Import thư viện tính toán mảng số học
import numpy as np
# Import framework tạo giao diện web Streamlit
import streamlit as st
# Import thư viện xử lý dữ liệu dạng bảng
import pandas as pd
# Import các hàm tiện ích từ module utils
from modules.utils import (
    bgr_to_rgb, rgb_to_ycrcb, plot_cr_cb_histogram, 
    plot_ycrcb_histogram
)
# HÀM PHÁT HIỆN VÙNG DA
def apply_skin_detection(image, cr_min, cr_max, cb_min, cb_max, morph_op, kernel_size):
    ycrcb = rgb_to_ycrcb(image)
    lower = np.array([0, cr_min, cb_min], dtype=np.uint8)
    upper = np.array([255, cr_max, cb_max], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if morph_op == "Opening":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    elif morph_op == "Closing":
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    elif morph_op == "Both":
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    skin_only = cv2.bitwise_and(image, image, mask=mask)
    # Trả về mặt nạ, ảnh vùng da, và ảnh YCrCb (để vẽ histogram)
    return mask, skin_only, ycrcb

def render_skin_detection():
    # Tiêu đề chính của trang
    st.header("1. Skin Detection - YCrCb Thresholding")
    with st.sidebar:
        # Tiêu đề phụ trong sidebar
        st.subheader("Tham so Skin Detection")
        
        # Widget upload file ảnh
        # key="skin": ID duy nhất để phân biệt với uploader ở module khác
        uploaded_file = st.file_uploader("Chon anh", type=['jpg', 'jpeg', 'png'], key="skin")
        
        # Slider điều chỉnh ngưỡng Cr (Red Chrominance)
        # Mặc định: 133-173 (từ nghiên cứu)
        cr_min = st.slider("Cr Min", 0, 255, 133, key="cr_min")  # Ngưỡng dưới Cr
        cr_max = st.slider("Cr Max", 0, 255, 173, key="cr_max")  # Ngưỡng trên Cr
        cb_min = st.slider("Cb Min", 0, 255, 77, key="cb_min")   # Ngưỡng dưới Cb
        cb_max = st.slider("Cb Max", 0, 255, 127, key="cb_max")  # Ngưỡng trên Cb
        morph_op = st.selectbox("Morphological", ["None", "Opening", "Closing", "Both"])

        kernel_size = st.slider("Kernel Size", 3, 15, 5, step=2)
    # Chỉ xử lý khi có file được upload
    if uploaded_file is not None:
        # Đọc file thành numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Giải mã thành ảnh OpenCV (BGR format)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Gọi hàm phát hiện da với các tham số đã chọn
        mask, skin_only, ycrcb = apply_skin_detection(
            image, cr_min, cr_max, cb_min, cb_max, morph_op, kernel_size
        )
        # Tạo 3 cột có chiều rộng bằng nhau
        col1, col2, col3 = st.columns(3)
        # Cột 1: Ảnh gốc
        with col1:
            st.markdown("**Anh goc**")
            # bgr_to_rgb: Chuyển BGR sang RGB để hiển thị đúng màu
            st.image(bgr_to_rgb(image), use_column_width=True)
        # Cột 2: Mặt nạ nhị phân
        with col2:
            st.markdown("**Binary Mask**")
            # Mask đã là grayscale, không cần chuyển đổi
            st.image(mask, use_column_width=True)
        # Cột 3: Vùng da đã tách
        with col3:
            st.markdown("**Vung da**")
            st.image(bgr_to_rgb(skin_only), use_column_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Mã hóa mask thành PNG và tạo nút download
            _, buffer = cv2.imencode('.png', mask)
            st.download_button("Tai Mask", buffer.tobytes(), "skin_mask.png", "image/png")
            
        with col2:
            # Mã hóa ảnh vùng da thành PNG và tạo nút download
            _, buffer = cv2.imencode('.png', skin_only)
            st.download_button("Tai Vung da", buffer.tobytes(), "skin_only.png", "image/png")
        st.subheader("Nguong Cr-Cb")
        
        # Hiển thị thông tin ngưỡng trong 2 cột
        threshold_col1, threshold_col2 = st.columns(2)
        
        with threshold_col1:
            # Hiển thị ngưỡng Cr với hộp thông tin màu xanh
            st.info(f"""
            **Nguong Cr (Red Chrominance):**
            - Min: {cr_min}
            - Max: {cr_max}
            - Range: {cr_max - cr_min}
            """)
            
        with threshold_col2:
            # Hiển thị ngưỡng Cb với hộp thông tin màu xanh
            st.info(f"""
            **Nguong Cb (Blue Chrominance):**
            - Min: {cb_min}
            - Max: {cb_max}
            - Range: {cb_max - cb_min}
            """)
        st.subheader("Histogram Y-Cr-Cb")
        st.markdown("**Histogram toan bo anh:**")
        # Đường kẻ đứng hiển thị ngưỡng Cr/Cb đã chọn
        fig_full = plot_ycrcb_histogram(
            ycrcb, 
            mask=None,  # Không lọc theo mask
            cr_min=cr_min, cr_max=cr_max, 
            cb_min=cb_min, cb_max=cb_max
        )
        # Hiển thị biểu đồ matplotlib
        st.pyplot(fig_full)
        st.markdown("**Histogram vung da (skin region):**")
        # Vẽ histogram CHỈ cho vùng da (sử dụng mask để lọc)
        fig_skin = plot_ycrcb_histogram(
            ycrcb, 
            mask=mask,  # Chỉ tính pixel trong vùng da
            cr_min=cr_min, cr_max=cr_max, 
            cb_min=cb_min, cb_max=cb_max
        )
        st.pyplot(fig_skin)
        
