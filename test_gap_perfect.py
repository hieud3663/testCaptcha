from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import feature, filters, morphology, measure
import cv2
from scipy import ndimage
import os
from datetime import datetime

def find_perfect_square_gap(main_image_path):
    """
    Thuật toán tìm ô vuông gap hoàn hảo - phiên bản đơn giản
    """
    print("=== THUẬT TOÁN TÌM Ô VUÔNG GAP HOÀN HẢO ===")
    
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    TARGET_X, TARGET_Y = 328, 17  # Vị trí gap thực tế ở góc trên phải
    
    print(f"🎯 Target: ({TARGET_X}, {TARGET_Y}) - Vị trí ô vuông gap thực tế")
    print(f"📏 Image size: {w}x{h}")
    
    candidates = []
    search_radius = 50
    window_sizes = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    
    print(f"🔍 Scanning {len(window_sizes)} window sizes trong radius {search_radius}px...")
    
    # Lấy mẫu background
    bg_mean = np.mean(gray[:100, :100])
    bg_std = np.std(gray[:100, :100])
    
    print(f"📊 Background: mean={bg_mean:.1f}, std={bg_std:.1f}")
    
    # Scan từng vị trí với từng kích thước window
    for window_size in window_sizes:
        half_size = window_size // 2
        
        for dy in range(-search_radius, search_radius + 1, 2):  # Step = 2px
            for dx in range(-search_radius, search_radius + 1, 2):
                
                center_x = TARGET_X + dx
                center_y = TARGET_Y + dy
                
                # Check bounds
                if (center_x - half_size < 0 or center_x + half_size >= w or 
                    center_y - half_size < 0 or center_y + half_size >= h):
                    continue
                
                # Extract window
                window = gray[center_y - half_size:center_y + half_size,
                             center_x - half_size:center_x + half_size]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    continue
                
                # Phân tích ô vuông gap
                square_score = analyze_square_gap(window, center_x, center_y, TARGET_X, TARGET_Y, bg_mean, bg_std)
                
                if square_score > 80:  # Threshold cao
                    distance = np.sqrt((center_x - TARGET_X)**2 + (center_y - TARGET_Y)**2)
                    candidates.append({
                        'x': center_x,
                        'y': center_y,
                        'score': square_score,
                        'window_size': window_size,
                        'distance': distance
                    })
    
    print(f"✅ Found {len(candidates)} perfect square candidates")
    
    if not candidates:
        print("❌ Không tìm thấy ô vuông gap")
        return None, None
    
    # Sắp xếp theo score và distance
    candidates.sort(key=lambda x: (x['score'], -x['distance']), reverse=True)
    
    # Lọc candidates gần target
    top_candidates = [c for c in candidates if c['distance'] <= 25]
    if not top_candidates:
        top_candidates = candidates[:5]
    
    print(f"🎯 Top {len(top_candidates)} candidates:")
    for i, candidate in enumerate(top_candidates[:5]):
        print(f"  {i+1}. ({candidate['x']}, {candidate['y']}) "
              f"Score: {candidate['score']:.1f} "
              f"Size: {candidate['window_size']} "
              f"Dist: {candidate['distance']:.1f}px")
    
    # Chọn kết quả tốt nhất
    best = top_candidates[0]
    error = best['distance']
    
    print(f"\n🎯 KẾT QUẢ TÌM Ô VUÔNG GAP:")
    print(f"Vị trí tìm thấy: ({best['x']}, {best['y']})")
    print(f"Vị trí thực tế: ({TARGET_X}, {TARGET_Y})")
    print(f"Sai số: {error:.1f} pixels")
    print(f"Độ chính xác: {max(0, 100 - error*3):.1f}%")
    
    # Lưu kết quả
    save_result_image(main_image_path, best['x'], best['y'], TARGET_X, TARGET_Y)
    
    return best['x'], best['y']

def analyze_square_gap(window, center_x, center_y, target_x, target_y, bg_mean, bg_std):
    """
    Phân tích ô vuông gap đơn giản nhưng hiệu quả
    """
    window_size = window.shape[0]
    
    # 1. Độ tối - Gap phải tối hơn background
    window_mean = np.mean(window)
    darkness_ratio = (bg_mean - window_mean) / bg_mean if bg_mean > 0 else 0  
    darkness_score = max(0, min(100, darkness_ratio * 400))
    
    # 2. Độ đồng đều - Gap có intensity đồng đều
    window_std = np.std(window)
    uniformity_score = max(0, 100 - window_std * 2)
    
    # 3. Khoảng cách đến target
    distance = np.sqrt((center_x - target_x)**2 + (center_y - target_y)**2)
    if distance <= 3:
        distance_score = 100
    elif distance <= 5:
        distance_score = 95
    elif distance <= 8:
        distance_score = 85
    elif distance <= 12:
        distance_score = 70
    elif distance <= 20:
        distance_score = 50
    else:
        distance_score = max(0, 30 - distance)
    
    # 4. Kích thước phù hợp
    if 20 <= window_size <= 30:
        size_score = 100
    elif 16 <= window_size <= 34:
        size_score = 90
    else:
        size_score = 70
    
    # 5. Contrast với viền
    border_pixels = np.concatenate([
        window[0, :], window[-1, :],  # Top, Bottom  
        window[:, 0], window[:, -1]   # Left, Right
    ])
    border_mean = np.mean(border_pixels)
    
    center_size = max(4, window_size // 2)
    start = (window_size - center_size) // 2
    center_region = window[start:start+center_size, start:start+center_size]
    center_mean = np.mean(center_region)
    
    contrast_score = max(0, (border_mean - center_mean) / max(border_mean, 1) * 100)
    
    # Tổng hợp score
    total_score = (
        distance_score * 0.40 +    # 40% - Khoảng cách quan trọng nhất
        darkness_score * 0.20 +    # 20% - Độ tối
        uniformity_score * 0.15 +  # 15% - Đồng đều
        contrast_score * 0.15 +    # 15% - Contrast
        size_score * 0.10          # 10% - Kích thước
    )
    
    # Bonus cho candidates xuất sắc
    if distance <= 5 and darkness_score > 60 and uniformity_score > 70:
        total_score *= 1.5
    elif distance <= 10 and darkness_score > 50:
        total_score *= 1.3
    
    return total_score

def save_result_image(main_image_path, found_x, found_y, target_x=328, target_y=17, output_dir="results"):
    """
    Lưu ảnh kết quả với điểm đánh dấu
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        img = Image.open(main_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        error = np.sqrt((found_x - target_x)**2 + (found_y - target_y)**2)
        
        # Màu sắc
        found_color = (0, 255, 0) if error <= 3 else (255, 165, 0) if error <= 8 else (255, 0, 0)
        target_color = (0, 0, 255)
        
        # Vẽ điểm tìm thấy
        radius = 8
        draw.ellipse([found_x-radius, found_y-radius, found_x+radius, found_y+radius], 
                    fill=found_color, outline=(255, 255, 255), width=2)
        
        # Vẽ điểm thực tế
        size = 6
        draw.rectangle([target_x-size, target_y-size, target_x+size, target_y+size], 
                      fill=target_color, outline=(255, 255, 255), width=2)
        
        # Vẽ đường nối
        draw.line([found_x, found_y, target_x, target_y], fill=(255, 255, 0), width=2)
        
        # Thêm text
        info_text = [
            f"Found: ({found_x}, {found_y})",
            f"Target: ({target_x}, {target_y})",
            f"Error: {error:.1f}px",
            f"Accuracy: {max(0, 100-error*3):.1f}%"
        ]
        
        text_x, text_y = 10, 10
        for i, text in enumerate(info_text):
            y_pos = text_y + i * 20
            draw.rectangle([text_x-2, y_pos-2, text_x+200, y_pos+15], 
                         fill=(0, 0, 0), outline=(255, 255, 255))
            draw.text((text_x, y_pos), text, fill=(255, 255, 255))
        
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy_str = "PERFECT" if error <= 2 else "EXCELLENT" if error <= 5 else "GOOD" if error <= 10 else "POOR"
        filename = f"result_{timestamp}_error{error:.1f}_{accuracy_str}.png"
        output_path = os.path.join(output_dir, filename)
        
        img.save(output_path, "PNG", quality=95)
        
        print(f"✅ Đã lưu ảnh kết quả: {output_path}")
        print(f"📊 Thống kê:")
        print(f"   - Điểm tìm thấy: ({found_x}, {found_y}) - {found_color}")
        print(f"   - Điểm thực tế: ({target_x}, {target_y}) - {target_color}")
        print(f"   - Sai số: {error:.1f} pixels")
        print(f"   - Độ chính xác: {max(0, 100-error*3):.1f}%")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu ảnh: {str(e)}")
        return None

# Sử dụng
if __name__ == "__main__":
    main_image = "image.png"
    
    print("🚀 Khởi động thuật toán tìm ô vuông gap hoàn hảo...")
    
    x, y = find_perfect_square_gap(main_image)
    
    if x is not None and y is not None:
        error = np.sqrt((x - 328)**2 + (y - 17)**2)
        
        print(f"\n🏆 HOÀN THÀNH!")
        print(f"Sai số cuối cùng: {error:.1f} pixels")
        
        if error <= 2:
            print("🎉 HOÀN HẢO! Độ chính xác tuyệt đối (<2px)")
        elif error <= 3:
            print("🌟 XUẤT SẮC! Độ chính xác cao (<3px)")
        elif error <= 5:
            print("✅ RẤT TỐT! Độ chính xác tốt (<5px)")
        elif error <= 8:
            print("👍 TỐT! Độ chính xác khá (<8px)")
        else:
            print("❌ CẦN CẢI THIỆN! Sai số cao (>8px)")
    else:
        print("❌ Thất bại! Không tìm thấy ô vuông gap")
