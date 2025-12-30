# Import thư viện xử lý ảnh OpenCV
import cv2
# Import thư viện tính toán mảng số học
import numpy as np
# Import thư viện vẽ biểu đồ
import matplotlib.pyplot as plt

def load_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image

def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_ycrcb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)


def ycrcb_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)


def calculate_mse(original, processed):
    # Chuyển sang float64 để tránh tràn số khi tính bình phương
    original = original.astype(np.float64)
    processed = processed.astype(np.float64)
    
    # Tính MSE: trung bình của bình phương sai khác
    mse = np.mean((original - processed) ** 2)
    
    return mse


def plot_histogram(image, channels, title, colors):
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (channel, color) in enumerate(zip(channels, colors)):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        # Vẽ đường histogram
        ax.plot(hist, color=color, label=channel)
    # Thiết lập nhãn trục
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()  # Hiển thị chú thích
    ax.set_xlim([0, 256])  # Giới hạn trục x
    return fig
def plot_cr_cb_histogram(ycrcb_image, mask=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # Lấy giá trị Cr và Cb
    if mask is not None:
        # Chỉ lấy pixel trong vùng mask (vùng da)
        cr_values = ycrcb_image[:, :, 1][mask > 0]  # Kênh 1 = Cr
        cb_values = ycrcb_image[:, :, 2][mask > 0]  # Kênh 2 = Cb
    else:
        # Lấy tất cả pixel
        cr_values = ycrcb_image[:, :, 1].flatten()  # .flatten(): Chuyển 2D → 1D
        cb_values = ycrcb_image[:, :, 2].flatten()
    # Vẽ histogram dạng cột
    axes[0].hist(cr_values, bins=50, color='red', alpha=0.7, edgecolor='darkred')
    # Vẽ đường ngưỡng tham khảo
    axes[0].axvline(x=133, color='green', linestyle='--', label='Threshold 133')
    axes[0].axvline(x=173, color='green', linestyle='--', label='Threshold 173')
    axes[0].set_xlabel('Cr Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Cr Channel Distribution')
    axes[0].legend()
    axes[1].hist(cb_values, bins=50, color='blue', alpha=0.7, edgecolor='darkblue')
    axes[1].axvline(x=77, color='green', linestyle='--', label='Threshold 77')
    axes[1].axvline(x=127, color='green', linestyle='--', label='Threshold 127')
    axes[1].set_xlabel('Cb Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cb Channel Distribution')
    axes[1].legend()
    # Tự động điều chỉnh khoảng cách giữa các subplot
    plt.tight_layout()
    
    return fig


def plot_ycrcb_histogram(ycrcb_image, mask=None, cr_min=133, cr_max=173, cb_min=77, cb_max=127):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if mask is not None and np.any(mask > 0):
        y_values = ycrcb_image[:, :, 0][mask > 0]   # Kênh 0 = Y
        cr_values = ycrcb_image[:, :, 1][mask > 0]  # Kênh 1 = Cr  
        cb_values = ycrcb_image[:, :, 2][mask > 0]  # Kênh 2 = Cb
    else:
        # Lấy tất cả pixel
        y_values = ycrcb_image[:, :, 0].flatten()
        cr_values = ycrcb_image[:, :, 1].flatten()
        cb_values = ycrcb_image[:, :, 2].flatten()

    axes[0].hist(y_values, bins=50, color='gray', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Y Value (Luminance)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Y Channel - Luminance')
    axes[0].set_xlim([0, 255]) 

    axes[1].hist(cr_values, bins=50, color='red', alpha=0.7, edgecolor='darkred')
    # Đường ngưỡng min/max
    axes[1].axvline(x=cr_min, color='green', linestyle='--', linewidth=2, label=f'Min: {cr_min}')
    axes[1].axvline(x=cr_max, color='green', linestyle='--', linewidth=2, label=f'Max: {cr_max}')
    # Tô màu vùng trong ngưỡng
    axes[1].axvspan(cr_min, cr_max, alpha=0.2, color='green')
    axes[1].set_xlabel('Cr Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Cr Channel Distribution')
    axes[1].legend()
    axes[1].set_xlim([0, 255])
    axes[2].hist(cb_values, bins=50, color='blue', alpha=0.7, edgecolor='darkblue')
    axes[2].axvline(x=cb_min, color='orange', linestyle='--', linewidth=2, label=f'Min: {cb_min}')
    axes[2].axvline(x=cb_max, color='orange', linestyle='--', linewidth=2, label=f'Max: {cb_max}')
    axes[2].axvspan(cb_min, cb_max, alpha=0.2, color='orange')
    axes[2].set_xlabel('Cb Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Cb Channel Distribution')
    axes[2].legend()
    axes[2].set_xlim([0, 255])
    
    plt.tight_layout()
    
    return fig


