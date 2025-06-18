#!/usr/bin/env python3

from PIL import Image, ImageDraw
import numpy as np
import cv2
from datetime import datetime
import os

def compare_methods():
    """
    So sánh kết quả của template matching với các phương pháp cũ
    """
    print("=== SO SÁNH CÁC PHƯƠNG PHÁP TÌM GAP ===")
    
    img = Image.open('image.png').convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    print(f"📏 Image size: {w}x{h}")
    
    # Kết quả từ template matching (mới nhất)
    template_result = (239, 138)
    
    # Kết quả từ các phương pháp cũ (từ debug trước đó)
    old_methods = [
        ("Simple Darkness", find_gap_simple_darkness(gray)),
        ("Multi-criteria", find_gap_multi_criteria(gray)),
        ("Corner Detection", find_gap_corner_detection(gray)),
    ]
    
    # So sánh quality của từng kết quả
    print(f"\n🔍 COMPARISON RESULTS:")
    print(f"Template Matching: {template_result}")
    analyze_gap_quality(gray, template_result[0], template_result[1], "Template Matching")
    
    for method_name, result in old_methods:
        if result:
            print(f"\n{method_name}: {result}")
            analyze_gap_quality(gray, result[0], result[1], method_name)
        else:
            print(f"\n{method_name}: Failed to find gap")
    
    # Visualize all results
    visualize_comparison(img, template_result, old_methods)

def find_gap_simple_darkness(gray):
    """
    Phương pháp cũ: tìm vùng tối nhất
    """
    h, w = gray.shape
    best_score = float('inf')
    best_pos = None
    
    window_size = 24
    half_size = window_size // 2
    
    for y in range(half_size, h - half_size, 4):
        for x in range(half_size, w - half_size, 4):
            window = gray[y - half_size:y + half_size, x - half_size:x + half_size]
            darkness = np.mean(window)
            
            if darkness < best_score:
                best_score = darkness
                best_pos = (x, y)
    
    return best_pos

def find_gap_multi_criteria(gray):
    """
    Phương pháp cũ: multi-criteria scoring
    """
    h, w = gray.shape
    best_score = 0
    best_pos = None
    
    window_size = 24
    half_size = window_size // 2
    
    for y in range(half_size, h - half_size, 4):
        for x in range(half_size, w - half_size, 4):
            window = gray[y - half_size:y + half_size, x - half_size:x + half_size]
            
            # Multi-criteria scoring
            darkness = 255 - np.mean(window)
            uniformity = 100 - np.std(window)
            
            # Simple combined score
            score = darkness * 0.7 + uniformity * 0.3
            
            if score > best_score:
                best_score = score
                best_pos = (x, y)
    
    return best_pos

def find_gap_corner_detection(gray):
    """
    Phương pháp cũ: corner detection
    """
    # Harris corner detection
    corners = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
    
    # Tìm corners với response cao
    corner_threshold = 0.01 * corners.max()
    corner_locations = np.where(corners > corner_threshold)
    
    if len(corner_locations[0]) > 0:
        # Lấy corner đầu tiên (có thể không phải gap thực sự)
        y, x = corner_locations[0][0], corner_locations[1][0]
        return (x, y)
    
    return None

def analyze_gap_quality(gray, x, y, method_name):
    """
    Phân tích chất lượng của gap position
    """
    window_size = 24
    half_size = window_size // 2
    
    if (x - half_size < 0 or x + half_size >= gray.shape[1] or
        y - half_size < 0 or y + half_size >= gray.shape[0]):
        print(f"  ❌ {method_name}: Out of bounds")
        return
    
    window = gray[y - half_size:y + half_size, x - half_size:x + half_size]
    
    darkness = 255 - np.mean(window)
    uniformity = 100 - np.std(window)  
    mean_val = np.mean(window)
    std_val = np.std(window)
    
    print(f"  📊 {method_name}:")
    print(f"    Mean: {mean_val:.1f} | Std: {std_val:.1f}")
    print(f"    Darkness: {darkness:.1f} | Uniformity: {uniformity:.1f}")
    
    # Quality assessment
    if mean_val < 30 and std_val < 25:
        print(f"    ✅ High quality gap candidate")
    elif mean_val < 60 and std_val < 40:
        print(f"    ⚠️ Medium quality gap candidate")
    else:
        print(f"    ❌ Low quality gap candidate")

def visualize_comparison(img, template_result, old_methods):
    """
    Visualize tất cả kết quả để so sánh
    """
    draw = ImageDraw.Draw(img)
    
    colors = [
        (255, 0, 0),    # Đỏ - Template Matching
        (0, 255, 0),    # Xanh lá - Simple Darkness
        (0, 0, 255),    # Xanh dương - Multi-criteria
        (255, 255, 0),  # Vàng - Corner Detection
    ]
    
    # Template matching result
    x, y = template_result
    draw.ellipse([x-12, y-12, x+12, y+12], outline=colors[0], width=4)
    draw.text((x+15, y-15), "Template", fill=colors[0])
    
    # Other methods
    for i, (method_name, result) in enumerate(old_methods):
        if result:
            x, y = result
            color = colors[i+1]
            draw.ellipse([x-8, y-8, x+8, y+8], outline=color, width=3)
            draw.text((x+10, y+10), method_name.split()[0], fill=color)
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/method_comparison_{timestamp}.png"
    if not os.path.exists("results"):
        os.makedirs("results")
    img.save(output_path)
    
    print(f"\n✅ Saved method comparison: {output_path}")

if __name__ == "__main__":
    compare_methods()
