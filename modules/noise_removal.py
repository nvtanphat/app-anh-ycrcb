# Import thư viện xử lý ảnh OpenCV
import cv2
# Import thư viện tính toán mảng số học
import numpy as np
# Import framework tạo giao diện web Streamlit
import streamlit as st
# Import thư viện xử lý dữ liệu dạng bảng
import pandas as pd
# Import các hàm tiện ích từ module utils
from modules.utils import bgr_to_rgb, rgb_to_ycrcb, ycrcb_to_rgb, calculate_mse
def add_salt_pepper_noise(image, density=0.05):
    noisy = image.copy()
    h, w = image.shape[:2]
    num_salt = int(density * h * w / 2)
    num_pepper = int(density * h * w / 2)  
    salt_coords = [np.random.randint(0, h, num_salt), np.random.randint(0, w, num_salt)]
    noisy[salt_coords[0], salt_coords[1]] = 255
    pepper_coords = [np.random.randint(0, h, num_pepper), np.random.randint(0, w, num_pepper)]
    noisy[pepper_coords[0], pepper_coords[1]] = 0
    return noisy
def denoise_mean(image, kernel_size=3):
    ycrcb = rgb_to_ycrcb(image)
    # Tách thành 3 kênh riêng biệt
    y, cr, cb = cv2.split(ycrcb)
    # Áp dụng bộ lọc trung bình (Mean/Box Filter) CHỈ trên kênh Y
    # cv2.blur: Tính trung bình cộng trong vùng kernel_size x kernel_size
    y_filtered = cv2.blur(y, (kernel_size, kernel_size))
    # Ghép lại các kênh thành ảnh YCrCb
    ycrcb_filtered = cv2.merge([y_filtered, cr, cb])
    # Chuyển về RGB và trả về
    return ycrcb_to_rgb(ycrcb_filtered)
# BỘ LỌC TRUNG VỊ (MEDIAN FILTER)
def denoise_median(image, kernel_size=3):
    ycrcb = rgb_to_ycrcb(image)
    # Tách thành 3 kênh riêng biệt
    y, cr, cb = cv2.split(ycrcb)
    # Áp dụng bộ lọc trung vị CHỈ trên kênh Y
    # cv2.medianBlur: Lấy giá trị trung vị trong vùng kernel
    # kernel_size PHẢI là số lẻ
    y_filtered = cv2.medianBlur(y, kernel_size)
    # Ghép lại các kênh thành ảnh YCrCb
    ycrcb_filtered = cv2.merge([y_filtered, cr, cb])
    # Chuyển về RGB và trả về
    return ycrcb_to_rgb(ycrcb_filtered)
# BỘ LỌC TRUNG VỊ THÍCH ỨNG (ADAPTIVE MEDIAN FILTER - AMF)
def adaptive_median_filter_pixel(image, y, x, max_size):
    h, w = image.shape
    size = 3  # Bắt đầu với cửa sổ 3x3
    while size <= max_size:
        # Tính offset (nửa kích thước cửa sổ)
        offset = size // 2
        # Lấy vùng lân cận (xử lý biên)
        y_min = max(0, y - offset)
        y_max = min(h, y + offset + 1)
        x_min = max(0, x - offset)
        x_max = min(w, x + offset + 1)
        window = image[y_min:y_max, x_min:x_max]
        # Tính các giá trị thống kê
        z_min = int(np.min(window))    # Giá trị nhỏ nhất
        z_max = int(np.max(window))    # Giá trị lớn nhất
        z_med = int(np.median(window)) # Trung vị
        z_xy = int(image[y, x])        # Giá trị pixel hiện tại
        if z_min < z_med < z_max:
            if z_min < z_xy < z_max:
                return z_xy   # Giữ nguyên pixel gốc
            else:
                return z_med  # Thay bằng trung vị
        else:
            size += 2  # Tăng kích thước cửa sổ (3→5→7→...)
    
    # Nếu đạt max_size mà vẫn không tìm được → trả về trung vị
    return z_med

def denoise_amf(image, max_kernel_size=7):
    # Chuyển đổi từ BGR sang YCrCb
    ycrcb = rgb_to_ycrcb(image)
    # Tách thành 3 kênh riêng biệt
    y, cr, cb = cv2.split(ycrcb)
    # Tạo ảnh output
    h, w = y.shape
    y_filtered = np.zeros_like(y)
    # Áp dụng AMF cho từng pixel
    for i in range(h):
        for j in range(w):
            y_filtered[i, j] = adaptive_median_filter_pixel(y, i, j, max_kernel_size)
    # Ghép lại các kênh thành ảnh YCrCb
    ycrcb_filtered = cv2.merge([y_filtered, cr, cb])
    # Chuyển về RGB và trả về
    return ycrcb_to_rgb(ycrcb_filtered)
def render_noise_removal():
    # Tiêu đề chính của trang
    st.header("3. Salt & Pepper Noise Removal")
    with st.sidebar:
        # Tiêu đề phụ trong sidebar
        st.subheader("Tham so Noise Removal")
        # Widget upload file ảnh
        uploaded_file = st.file_uploader("Chon anh", type=['jpg', 'jpeg', 'png'], key="noise")
        # Slider chọn mật độ nhiễu (2% - 10%)
        # Chia cho 100 để chuyển từ % sang tỷ lệ (0.02 - 0.10)
        noise_density = st.slider("Mat do nhieu (%)", 2, 10, 5) / 100
        # Dropdown chọn kích thước kernel
        # index=0: Mặc định chọn giá trị đầu tiên (3)
        kernel_size = st.selectbox("Kernel Size", [3, 5, 7], index=0)
    if uploaded_file is not None:
        # Đọc file thành numpy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        # Giải mã thành ảnh OpenCV (BGR format)
        original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Bước 1: Thêm nhiễu Salt & Pepper vào ảnh gốc
        noisy = add_salt_pepper_noise(original, noise_density)
        # Bước 2: Áp dụng bộ lọc Trung bình (Mean Filter)
        mean_result = denoise_mean(noisy, kernel_size)
        # Bước 3: Áp dụng bộ lọc Trung vị (Median Filter)
        median_result = denoise_median(noisy, kernel_size)
        # Bước 4: Áp dụng bộ lọc Trung vị Thích ứng (AMF)
        amf_result = denoise_amf(noisy, max_kernel_size=7)
        # Hàng 1: Ảnh gốc và ảnh nhiễu
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Ảnh gốc**")
            # Chuyển BGR sang RGB để hiển thị đúng màu
            st.image(bgr_to_rgb(original), use_column_width=True)
            
        with col2:
            st.markdown("**Ảnh nhiễu**")
            st.image(bgr_to_rgb(noisy), use_column_width=True)
        # Hàng 2: So sánh 3 phương pháp lọc
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Mean Filter**")
            st.image(bgr_to_rgb(mean_result), use_column_width=True)  
        with col2:
            st.markdown("**Median Filter**")
            st.image(bgr_to_rgb(median_result), use_column_width=True)
            
        with col3:
            st.markdown("**AMF**")
            st.image(bgr_to_rgb(amf_result), use_column_width=True)
        # Tạo 3 cột cho 3 nút download
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Mã hóa ảnh Mean result thành PNG
            _, buffer = cv2.imencode('.png', mean_result)
            st.download_button(
                "Tai anh (Mean)",
                buffer.tobytes(),
                "mean_result.png",
                "image/png"
            )
            
        with col2:
            # Mã hóa ảnh Median result thành PNG
            _, buffer = cv2.imencode('.png', median_result)
            st.download_button(
                "Tai anh (Median)",
                buffer.tobytes(),
                "median_result.png",
                "image/png"
            )
        with col3:
            # Mã hóa ảnh AMF result thành PNG
            _, buffer = cv2.imencode('.png', amf_result)
            st.download_button(
                "Tai anh (AMF)",
                buffer.tobytes(),
                "amf_result.png",
                "image/png"
            )