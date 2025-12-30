import cv2
# Import thư viện tính toán mảng số học
import numpy as np
# Import framework tạo giao diện web Streamlit
import streamlit as st
# Import thư viện xử lý dữ liệu dạng bảng
import pandas as pd
# Import các hàm tiện ích từ module utils
from modules.utils import bgr_to_rgb, rgb_to_ycrcb, ycrcb_to_rgb


# ĐỊNH NGHĨA CÁC KERNEL LÀM NÉT
KERNELS = {
    "Laplacian": np.array([
        [0, -1, 0],      # Hàng trên: chỉ pixel phía trên có trọng số -1
        [-1, 5, -1],     # Hàng giữa: pixel trung tâm = 5, trái/phải = -1
        [0, -1, 0]       # Hàng dưới: chỉ pixel phía dưới có trọng số -1
    ], dtype=np.float32),  # Kiểu float32 để tính toán chính xác
    
    "Laplacian 8-dir": np.array([
        [-1, -1, -1],    # Hàng trên: tất cả 3 pixel đều -1
        [-1, 9, -1],     # Hàng giữa: trung tâm = 9, trái/phải = -1
        [-1, -1, -1]     # Hàng dưới: tất cả 3 pixel đều -1
    ], dtype=np.float32),
}


# HÀM TẠO KERNEL VỚI ĐỘ MẠNH TÙY CHỈNH
def get_kernel(kernel_type, strength=1.0):
    if kernel_type == "Laplacian":
        kernel = np.array([
            [0, -strength, 0],                        # Trên = -strength
            [-strength, 1 + 4*strength, -strength],   # Trái = -strength, Giữa = 1+4s, Phải = -strength
            [0, -strength, 0]                         # Dưới = -strength
        ], dtype=np.float32)
    else: 
        kernel = np.array([
            [-strength, -strength, -strength],            # 3 pixel trên
            [-strength, 1 + 8*strength, -strength],       # Giữa = 1+8s
            [-strength, -strength, -strength]             # 3 pixel dưới
        ], dtype=np.float32)
        
    return kernel

# HÀM LÀM NÉT CHỈ TRÊN KÊNH Y (PHƯƠNG PHÁP THÔNG MINH)
def sharpen_y_channel(image, kernel):

    ycrcb = rgb_to_ycrcb(image)
    y, cr, cb = cv2.split(ycrcb)
    y_sharpened = cv2.filter2D(y, -1, kernel)
    y_sharpened = np.clip(y_sharpened, 0, 255).astype(np.uint8)
    ycrcb_sharpened = cv2.merge([y_sharpened, cr, cb])
    return ycrcb_to_rgb(ycrcb_sharpened)

# HÀM RENDER GIAO DIỆN STREAMLIT

def render_smart_sharpening():
    # Tiêu đề chính của trang
    st.header("2. Smart Sharpening - Y-channel vs RGB")
    # SIDEBAR - CONTROLS
    # Context manager để đặt các widget vào sidebar
    with st.sidebar:
        # Tiêu đề phụ trong sidebar
        st.subheader("Tham so Sharpening")
        # Widget upload file ảnh
        # type: Chỉ chấp nhận file jpg, jpeg, png
        # key: ID duy nhất để Streamlit phân biệt với các uploader khác
        uploaded_file = st.file_uploader("Chon anh", type=['jpg', 'jpeg', 'png'], key="sharp")
        # Dropdown để chọn loại kernel
        # list(KERNELS.keys()) = ["Laplacian", "Laplacian 8-dir", "Unsharp Mask"]
        kernel_type = st.selectbox("Loai Kernel", list(KERNELS.keys()))
        # Slider điều chỉnh độ mạnh làm nét
        strength = st.slider("Strength", 0.5, 3.0, 1.0, 0.1, key="sharp_str")
        # Tạo kernel với loại và strength đã chọn
        kernel = get_kernel(kernel_type, strength)
        # Hiển thị ma trận kernel (làm tròn 2 chữ số thập phân)
        st.text(f"Kernel:\n{np.round(kernel, 2)}")
    # MAIN CONTENT - XỬ LÝ ẢNH
    # Chỉ xử lý khi có file được upload
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Tạo kernel với các tham số hiện tại
        kernel = get_kernel(kernel_type, strength)
        # Áp dụng làm nét trên kênh Y (bảo toàn màu sắc)
        y_result = sharpen_y_channel(image, kernel)
        # HIỂN THỊ ẢNH 2 CỘT
        col1, col2 = st.columns(2)
        # Cột 1: Ảnh gốc
        with col1:
            st.markdown("**Ảnh gốc**")
            st.image(bgr_to_rgb(image), use_column_width=True)
        # Cột 2: Làm nét Y-channel
        with col2:
            st.markdown(f"**Y-channel ({kernel_type})**")
            st.image(bgr_to_rgb(y_result), use_column_width=True)
        # NÚT TẢI XUỐNG
        _, buffer_y = cv2.imencode('.png', y_result)
        st.download_button(
            label=f"Tải ảnh đã làm nét",
            data=buffer_y.tobytes(),
            file_name="sharpened_y_channel.png",
            mime="image/png"
        )