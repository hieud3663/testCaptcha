#!/usr/bin/env python3
"""
PHÂN TÍCH CAPTCHA để hiểu cấu trúc và tối ưu thuật toán
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_captcha(image_path):
    """Phân tích cấu trúc captcha"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"📐 Kích thước ảnh: {w}x{h}")
    
    # Hiển thị ảnh gốc
    plt.figure(figsize=(15, 10))
    
    # Ảnh gốc
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Ảnh gốc')
    plt.axis('off')
    
    # Ảnh xám
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Ảnh xám')
    plt.axis('off')
    
    # Phân vùng theo chiều dọc
    upper_part = gray[:int(h*0.6), :]
    middle_part = gray[int(h*0.6):int(h*0.8), :]
    lower_part = gray[int(h*0.8):, :]
    
    plt.subplot(2, 3, 3)
    plt.imshow(upper_part, cmap='gray')
    plt.title('Vùng trên (60%)')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(middle_part, cmap='gray')
    plt.title('Vùng giữa (60-80%)')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(lower_part, cmap='gray')
    plt.title('Vùng dưới (80%+)')
    plt.axis('off')
    
    # Phân tích histogram
    plt.subplot(2, 3, 6)
    plt.hist(gray.ravel(), 256, [0, 256], alpha=0.7, label='Toàn bộ')
    plt.hist(upper_part.ravel(), 256, [0, 256], alpha=0.7, label='Vùng trên')
    plt.hist(lower_part.ravel(), 256, [0, 256], alpha=0.7, label='Vùng dưới')
    plt.legend()
    plt.title('Histogram')
    
    plt.tight_layout()
    plt.savefig('results/captcha_analysis.png', dpi=150)
    plt.show()
    
    # Phân tích chi tiết vùng dưới
    print("\n🔍 PHÂN TÍCH VÙNG DƯỚI:")
    print(f"Kích thước vùng dưới: {lower_part.shape}")
    print(f"Mean: {np.mean(lower_part):.1f}")
    print(f"Std: {np.std(lower_part):.1f}")
    print(f"Min: {np.min(lower_part)}")
    print(f"Max: {np.max(lower_part)}")
    
    # Thử tìm mảnh ghép với nhiều phương pháp
    find_piece_candidates(img_rgb, gray)

def find_piece_candidates(img_rgb, gray):
    """Tìm các ứng viên mảnh ghép"""
    h, w = gray.shape
    
    print("\n🧩 TÌM MẢNH GHÉP:")
    
    # Thử nhiều vùng khác nhau
    regions = [
        ("Vùng dưới 25%", gray[int(h*0.75):, :], int(h*0.75)),
        ("Vùng dưới 20%", gray[int(h*0.8):, :], int(h*0.8)),
        ("Vùng dưới 15%", gray[int(h*0.85):, :], int(h*0.85)),
        ("Vùng dưới 30%", gray[int(h*0.7):, :], int(h*0.7)),
    ]
    
    for region_name, region, offset_y in regions:
        print(f"\n--- {region_name} ---")
        print(f"Kích thước: {region.shape}")
        
        # Thử nhiều threshold
        for thresh in [50, 80, 100, 120, 150, 200]:
            _, binary = cv2.threshold(region, thresh, 255, cv2.THRESH_BINARY)
            
            # Tìm contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Lọc noise
                    x, y, cw, ch = cv2.boundingRect(contour)
                    aspect_ratio = cw / ch if ch > 0 else 0
                    valid_contours.append((area, aspect_ratio, (x, y + offset_y, cw, ch)))
            
            if valid_contours:
                print(f"  Threshold {thresh}: {len(valid_contours)} contours")
                # Sắp xếp theo diện tích
                valid_contours.sort(reverse=True)
                for i, (area, ratio, bbox) in enumerate(valid_contours[:3]):
                    print(f"    {i+1}. Area: {area:.0f}, Ratio: {ratio:.2f}, Bbox: {bbox}")
    
    # Visualize các threshold khác nhau cho vùng dưới
    bottom_region = gray[int(h*0.8):, :]
    plt.figure(figsize=(15, 10))
    
    thresholds = [50, 80, 100, 120, 150, 200]
    for i, thresh in enumerate(thresholds):
        _, binary = cv2.threshold(bottom_region, thresh, 255, cv2.THRESH_BINARY)
        
        plt.subplot(2, 3, i+1)
        plt.imshow(binary, cmap='gray')
        plt.title(f'Threshold {thresh}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/threshold_analysis.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    analyze_captcha("image.png")
