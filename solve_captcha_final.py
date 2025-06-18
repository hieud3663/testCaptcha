#!/usr/bin/env python3
"""
GIẢI CAPTCHA PUZZLE - PHIÊN BẢN TỐI ƯU
Nhận diện hình ảnh để tìm vị trí ghép mảnh chính xác
"""

import cv2
import numpy as np
from datetime import datetime
import os

def solve_captcha_puzzle(image_path):
    """Giải captcha puzzle thông minh"""
    print("🎯 BẮT ĐẦU GIẢI CAPTCHA PUZZLE...")
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"📐 Kích thước ảnh: {w}x{h}")
    
    # Phân tích ảnh trước
    analyze_image_structure(gray)
    
    # Bước 1: Tìm mảnh ghép (thử nhiều cách)
    piece_info = find_puzzle_piece_smart(img, gray)
    if piece_info is None:
        print("❌ Không tìm thấy mảnh ghép!")
        return None
    
    piece_template, piece_bbox = piece_info
    print(f"🧩 Đã tìm thấy mảnh ghép: {piece_bbox}")
    
    # Bước 2: Tìm vị trí khớp
    gap_position = find_best_match_position(gray, piece_template, piece_bbox)
    
    if gap_position is None:
        print("❌ Không tìm thấy vị trí khớp!")
        return None
    
    # Bước 3: Visualize và lưu kết quả
    save_result_image(img, piece_bbox, gap_position)
    
    print(f"✅ THÀNH CÔNG! Vị trí cần click: {gap_position}")
    return gap_position

def analyze_image_structure(gray):
    """Phân tích cấu trúc ảnh"""
    h, w = gray.shape
    
    # Phân tích các vùng
    regions = {
        'Toàn bộ': gray,
        'Vùng trên (70%)': gray[:int(h*0.7), :],
        'Vùng dưới (70%+)': gray[int(h*0.7):, :],
        'Vùng dưới (80%+)': gray[int(h*0.8):, :],
        'Vùng dưới (85%+)': gray[int(h*0.85):, :],
    }
    
    print("\n📊 PHÂN TÍCH CẤU TRÚC:")
    for name, region in regions.items():
        if region.size > 0:
            print(f"{name}: {region.shape}, Mean={np.mean(region):.1f}, Std={np.std(region):.1f}")

def find_puzzle_piece_smart(img, gray):
    """Tìm mảnh ghép bằng nhiều phương pháp thông minh"""
    h, w = gray.shape
    
    print("\n🔍 TÌM MẢNH GHÉP...")
    
    # Thử nhiều vùng tìm kiếm
    search_regions = [
        (0.75, "25% dưới"),
        (0.8, "20% dưới"), 
        (0.85, "15% dưới"),
        (0.7, "30% dưới"),
        (0.65, "35% dưới")
    ]
    
    best_piece = None
    best_score = 0
    
    for start_ratio, region_name in search_regions:
        start_y = int(h * start_ratio)
        region = gray[start_y:, :]
        
        print(f"\n🔍 Tìm trong {region_name}...")
        
        # Thử nhiều phương pháp threshold
        methods = [
            ('THRESH_BINARY', cv2.THRESH_BINARY),
            ('THRESH_BINARY_INV', cv2.THRESH_BINARY_INV),
            ('THRESH_OTSU', cv2.THRESH_BINARY + cv2.THRESH_OTSU),
        ]
        
        for method_name, thresh_type in methods:
            # Thử với nhiều giá trị threshold
            thresh_values = [0] if 'OTSU' in method_name else [50, 80, 100, 120, 150, 180, 200]
            
            for thresh_val in thresh_values:
                try:
                    if 'OTSU' in method_name:
                        _, binary = cv2.threshold(region, 0, 255, thresh_type)
                    else:
                        _, binary = cv2.threshold(region, thresh_val, 255, thresh_type)
                    
                    # Morphology để làm sạch
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                    
                    # Tìm contours
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        
                        # Mảnh ghép phải có kích thước hợp lý
                        if 200 < area < 8000:
                            x, y, cw, ch = cv2.boundingRect(contour)
                            
                            # Kiểm tra tỷ lệ khung hình hợp lý
                            aspect_ratio = cw / ch if ch > 0 else 0
                            if 0.3 < aspect_ratio < 3.0:
                                
                                # Điều chỉnh tọa độ về ảnh gốc
                                abs_y = start_y + y
                                abs_bbox = (x, abs_y, cw, ch)
                                
                                # Extract template
                                template = gray[abs_y:abs_y+ch, x:x+cw]
                                
                                # Đánh giá chất lượng mảnh ghép
                                score = evaluate_piece_quality(template, area, aspect_ratio)
                                
                                if score > best_score:
                                    best_score = score
                                    best_piece = (template, abs_bbox)
                                    print(f"✨ Mảnh ghép tốt hơn: {method_name}, thresh={thresh_val}, area={area:.0f}, ratio={aspect_ratio:.2f}, score={score:.2f}")
                
                except Exception as e:
                    continue
    
    if best_piece:
        template, bbox = best_piece
        print(f"🏆 Chọn mảnh ghép tốt nhất: {bbox}, score={best_score:.2f}")
        return best_piece
    
    return None

def evaluate_piece_quality(template, area, aspect_ratio):
    """Đánh giá chất lượng mảnh ghép"""
    if template.size == 0:
        return 0
    
    # Tiêu chí đánh giá:
    # 1. Kích thước hợp lý (ưu tiên medium size)
    size_score = 1.0 - abs(area - 2000) / 5000  # Tối ưu quanh 2000 pixels
    size_score = max(0, min(1, size_score))
    
    # 2. Tỷ lệ khung hình (ưu tiên hình vuông hoặc hình chữ nhật cân đối)
    aspect_score = 1.0 - abs(aspect_ratio - 1.0) / 2.0  # Tối ưu quanh 1.0
    aspect_score = max(0, min(1, aspect_score))
    
    # 3. Độ phức tạp (có chi tiết, không đồng nhất)
    std_dev = np.std(template)
    complexity_score = min(1.0, std_dev / 50.0)  # Chuẩn hóa về [0,1]
    
    # 4. Độ tương phản
    contrast = np.max(template) - np.min(template)
    contrast_score = min(1.0, contrast / 255.0)
    
    # Tổng hợp điểm
    total_score = (size_score * 0.3 + aspect_score * 0.2 + 
                   complexity_score * 0.3 + contrast_score * 0.2)
    
    return total_score

def find_best_match_position(gray, piece_template, piece_bbox):
    """Tìm vị trí khớp tốt nhất"""
    h, w = gray.shape
    px, py, pw, ph = piece_bbox
    
    print(f"\n🎯 TÌM VỊ TRÍ KHỚP cho mảnh {pw}x{ph}...")
    
    # Vùng tìm kiếm: phần trên, tránh vùng mảnh ghép
    search_end_y = max(10, py - 20)  # Tránh overlap
    search_region = gray[:search_end_y, :]
    
    if search_region.size == 0:
        print("❌ Vùng tìm kiếm quá nhỏ!")
        return None
    
    print(f"🔍 Vùng tìm kiếm: {search_region.shape}")
    
    # Template matching với nhiều phương pháp
    methods = [
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED, False),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED, False),
        ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED, True),  # True = take minimum
    ]
    
    best_position = None
    best_confidence = 0
    
    # Thử với template gốc và các biến thể
    templates_to_try = [
        ("Original", piece_template),
        ("Inverted", 255 - piece_template),
    ]
    
    for template_name, template in templates_to_try:
        # Đảm bảo template không lớn hơn search region
        if template.shape[0] >= search_region.shape[0] or template.shape[1] >= search_region.shape[1]:
            continue
            
        print(f"\n🔍 Thử template: {template_name}")
        
        for method_name, method, take_min in methods:
            try:
                result = cv2.matchTemplate(search_region, template, method)
                
                if take_min:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    confidence = 1 - min_val  # Đảo ngược cho SQDIFF
                    match_loc = min_loc
                else:
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    confidence = max_val
                    match_loc = max_loc
                
                # Tính vị trí center của match
                center_x = match_loc[0] + template.shape[1] // 2
                center_y = match_loc[1] + template.shape[0] // 2
                
                print(f"  {method_name}: confidence={confidence:.3f} at ({center_x}, {center_y})")
                
                # Xác thực thêm nếu confidence cao
                if confidence > 0.3:
                    # Kiểm tra vùng xung quanh có phù hợp không
                    validation_score = validate_gap_region(search_region, match_loc, template.shape)
                    total_score = confidence * 0.7 + validation_score * 0.3
                    
                    if total_score > best_confidence:
                        best_confidence = total_score
                        best_position = (center_x, center_y)
                        print(f"    🎯 Vị trí tốt hơn! Score: {total_score:.3f}")
            
            except Exception as e:
                print(f"    ❌ Lỗi {method_name}: {e}")
                continue
    
    if best_position:
        print(f"\n🏆 VỊ TRÍ TỐT NHẤT: {best_position} (confidence: {best_confidence:.3f})")
        return best_position
    
    return None

def validate_gap_region(search_region, match_loc, template_shape):
    """Xác thực vùng gap có phù hợp không"""
    try:
        mx, my = match_loc
        th, tw = template_shape
        
        # Đảm bảo không vượt biên
        if mx + tw > search_region.shape[1] or my + th > search_region.shape[0]:
            return 0
        
        gap_region = search_region[my:my+th, mx:mx+tw]
        
        if gap_region.size == 0:
            return 0
        
        # Đánh giá đặc trưng gap:
        # 1. Độ đồng đều (gap thường có màu đồng đều)
        uniformity = 1.0 / (1.0 + np.std(gap_region) / 50.0)
        
        # 2. Độ tối (gap thường tối hơn background)
        darkness = 1.0 - np.mean(gap_region) / 255.0
        
        # 3. Độ tương phản thấp (gap thường ít chi tiết)
        contrast = (np.max(gap_region) - np.min(gap_region)) / 255.0
        low_contrast = 1.0 - contrast
        
        # Tổng hợp
        score = (uniformity * 0.4 + darkness * 0.4 + low_contrast * 0.2)
        return max(0, min(1, score))
    
    except:
        return 0

def save_result_image(img, piece_bbox, gap_position):
    """Lưu ảnh kết quả"""
    result_img = img.copy()
    px, py, pw, ph = piece_bbox
    gx, gy = gap_position
    
    # Vẽ mảnh ghép
    cv2.rectangle(result_img, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
    cv2.putText(result_img, 'PIECE', (px, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Vẽ vị trí gap
    cv2.circle(result_img, gap_position, 8, (0, 255, 0), -1)
    cv2.circle(result_img, gap_position, 12, (0, 255, 0), 2)
    cv2.putText(result_img, f'GAP', (gx + 15, gy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Vẽ đường nối
    piece_center = (px + pw//2, py + ph//2)
    cv2.arrowedLine(result_img, piece_center, gap_position, (255, 255, 0), 2)
    
    # Lưu file
    if not os.path.exists('results'):
        os.makedirs('results')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/captcha_solved_{timestamp}.png"
    cv2.imwrite(result_path, result_img)
    
    print(f"💾 Đã lưu kết quả: {result_path}")

def main():
    """Hàm main"""
    image_path = "image.png"
    
    try:
        gap_position = solve_captcha_puzzle(image_path)
        if gap_position:
            print(f"\n🎉 HOÀN THÀNH!")
            print(f"📍 Vị trí cần click: {gap_position}")
            print(f"💡 Tọa độ X: {gap_position[0]}, Y: {gap_position[1]}")
        else:
            print("\n❌ THẤT BẠI - Không tìm thấy vị trí phù hợp")
    
    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
