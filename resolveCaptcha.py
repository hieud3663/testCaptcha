from PIL import Image
import numpy as np
import cv2
import os
import sys
import traceback
import tempfile
import base64
import io
import json
from datetime import datetime
from skimage import feature, filters, morphology, measure
from skimage.segmentation import flood_fill
from scipy import ndimage
from skimage.feature import peak_local_max
# Bỏ import match_template vì không tồn tại
# from skimage.transform import match_template

# Đã di chuyển các thuật toán mới xuống dưới

def cv2_template_matching(search_area, template):
    """
    Sử dụng cv2 để template matching thay thế cho skimage
    """
    # Ensure both arrays are float32
    search_area_f = search_area.astype(np.float32)
    template_f = template.astype(np.float32)
    
    # Template matching
    result = cv2.matchTemplate(search_area_f, template_f, cv2.TM_CCOEFF_NORMED)
    
    # Find all good matches (threshold = 0.3)
    locations = np.where(result >= 0.3)
    matches = []
    
    for pt in zip(*locations[::-1]):  # Switch columns and rows
        confidence = result[pt[1], pt[0]]
        matches.append((pt[0], pt[1], confidence))
    
    return matches

def find_gap_ultra_precise(main_image_path, piece_info):
    """
    Tìm gap với độ chính xác siêu cao
    """
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    
    # Vùng tìm kiếm được tối ưu hóa
    search_top = 0
    search_bottom = h // 2  # Chỉ tìm nửa trên
    search_left = w // 8    # Bỏ qua viền trái
    search_right = w * 7 // 8  # Bỏ qua viền phải
    
    search_area = gray[search_top:search_bottom, search_left:search_right]
    search_h, search_w = search_area.shape
    
    print(f"Vùng tìm kiếm: {search_w}x{search_h}")
    
    # Tham số từ mảnh ghép
    target_area = piece_info['area']
    target_width = piece_info['width']
    target_height = piece_info['height']
    
    # Tolerance chặt chẽ hơn
    area_tolerance = 0.25  # ±25%
    min_area = target_area * (1 - area_tolerance)
    max_area = target_area * (1 + area_tolerance)
    
    print(f"Tìm vùng gap: {min_area:.0f} - {max_area:.0f} pixels")
    
    all_candidates = []
    
    # === PHƯƠNG PHÁP 1: MULTI-THRESHOLD ANALYSIS ===
    thresholds = [
        np.mean(search_area) - 1.5 * np.std(search_area),
        np.mean(search_area) - 1.0 * np.std(search_area),
        np.mean(search_area) - 0.5 * np.std(search_area),
        np.mean(search_area),
    ]
    
    for i, threshold in enumerate(thresholds):
        binary = search_area < threshold
        
        # Morphological operations
        kernel = morphology.disk(2)
        binary_clean = morphology.opening(binary, kernel)
        binary_clean = morphology.closing(binary_clean, morphology.disk(3))
        
        # Connected components
        labeled = measure.label(binary_clean)
        regions = measure.regionprops(labeled)
        
        for region in regions:
            if min_area <= region.area <= max_area:
                # Tính các đặc trưng
                bbox = region.bbox
                region_width = bbox[3] - bbox[1]
                region_height = bbox[2] - bbox[0]
                
                if region_height == 0:
                    continue
                    
                region_aspect = region_width / region_height
                
                # Kiểm tra aspect ratio
                aspect_diff = abs(region_aspect - piece_info['aspect_ratio'])
                if aspect_diff < 0.6:  # Chặt chẽ hơn
                    centroid_y, centroid_x = region.centroid
                    
                    # Multi-criteria scoring
                    area_score = 1 - abs(region.area - target_area) / target_area
                    aspect_score = 1 - aspect_diff / 0.6
                    eccentricity_score = 1 - region.eccentricity
                    extent_score = region.extent
                    
                    total_score = (
                        area_score * 100 +
                        aspect_score * 80 +
                        eccentricity_score * 30 +
                        extent_score * 20
                    )
                    
                    global_x = int(centroid_x + search_left)
                    global_y = int(centroid_y + search_top)
                    
                    all_candidates.append({
                        'x': global_x,
                        'y': global_y,
                        'score': total_score,
                        'area': region.area,
                        'method': f'threshold_{i}',
                        'details': {
                            'area_score': area_score,
                            'aspect_score': aspect_score,
                            'eccentricity': region.eccentricity,
                            'extent': region.extent
                        }
                    })
    
    # === PHƯƠNG PHÁP 2: ADVANCED EDGE-BASED ANALYSIS ===
    # Multiple edge detection methods
    edges_canny = feature.canny(search_area, sigma=1.0, low_threshold=0.05, high_threshold=0.15)
    edges_sobel = filters.sobel(search_area) > 0.1
    edges_scharr = filters.scharr(search_area) > 0.1
    edges_prewitt = filters.prewitt(search_area) > 0.1
    
    # Combine edges
    edges_combined = edges_canny | edges_sobel | edges_scharr | edges_prewitt
    
    # Morphological processing
    kernel = morphology.disk(2)
    edges_processed = morphology.closing(edges_combined, kernel)
    edges_processed = morphology.opening(edges_processed, morphology.disk(1))
    
    # Find contours
    contours, _ = cv2.findContours(edges_processed.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            # Tính centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Tính các đặc trưng
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Bounding box
                x, y, w_cont, h_cont = cv2.boundingRect(contour)
                aspect_ratio = w_cont / h_cont if h_cont > 0 else 1
                
                # Convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Scoring
                area_score = 1 - abs(area - target_area) / target_area
                aspect_score = 1 - abs(aspect_ratio - piece_info['aspect_ratio'])
                circ_score = 1 - abs(circularity - piece_info['circularity'])
                solid_score = abs(solidity - piece_info['solidity'])  # Gap should have different solidity
                
                total_score = (
                    area_score * 120 +
                    aspect_score * 60 +
                    circ_score * 40 +
                    solid_score * 30
                )
                
                global_x = cx + search_left
                global_y = cy + search_top
                
                all_candidates.append({
                    'x': global_x,
                    'y': global_y,
                    'score': total_score,
                    'area': area,
                    'method': 'edge_contour',
                    'details': {
                        'area_score': area_score,
                        'aspect_score': aspect_score,
                        'circularity': circularity,
                        'solidity': solidity
                    }
                })
    
    # === PHƯƠNG PHÁP 3: TEMPLATE MATCHING MULTI-SCALE ===
    scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    
    for scale in scales:
        template_size = int(np.sqrt(target_area) * scale)
        if template_size < 10 or template_size > min(search_h, search_w) // 2:
            continue
            
        template = create_precise_gap_template(template_size, piece_info)
        
        # Template matching sử dụng cv2
        matches = cv2_template_matching(search_area, template)
        
        for match_x, match_y, confidence in matches:
            if confidence > 0.3:  # Threshold
                # Convert to global coordinates
                global_x = match_x + template_size // 2 + search_left
                global_y = match_y + template_size // 2 + search_top
                
                # Estimate area
                estimated_area = template_size * template_size * 0.8
                area_score = 1 - abs(estimated_area - target_area) / target_area
                
                total_score = confidence * 150 + area_score * 50
                
                all_candidates.append({
                    'x': global_x,
                    'y': global_y,
                    'score': total_score,
                    'area': estimated_area,
                    'method': f'template_{scale:.1f}',
                    'details': {
                        'confidence': confidence,
                        'area_score': area_score,
                        'template_size': template_size
                    }
                })
    
    # === ENSEMBLE VÀ FILTERING ===
    if all_candidates:
        print(f"Tổng cộng {len(all_candidates)} candidates từ tất cả phương pháp")
        
        # Sắp xếp theo score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Advanced Non-Maximum Suppression
        final_candidates = advanced_nms(all_candidates, 
                                       distance_threshold=25, 
                                       score_threshold=30)
        
        # Ensemble scoring
        ensemble_candidates = ensemble_scoring(final_candidates)
        
        print(f"Sau filtering: {len(ensemble_candidates)} candidates")
        
        # In top candidates
        print("\nTop 10 candidates:")
        for i, candidate in enumerate(ensemble_candidates[:10]):
            error = np.sqrt((candidate['x'] - 275)**2 + (candidate['y'] - 26)**2)
            print(f"  {i+1}. ({candidate['x']}, {candidate['y']}) "
                  f"- Score: {candidate['score']:.1f} "
                  f"- Method: {candidate['method']} "
                  f"- Error: {error:.1f}")
        
        # Chọn kết quả tốt nhất
        best_candidate = ensemble_candidates[0]
        return best_candidate['x'], best_candidate['y']
    
    return None, None

def create_precise_gap_template(size, piece_info):
    """
    Tạo template chính xác dựa trên thông tin mảnh ghép
    """
    template = np.ones((size, size)) * 128  # Background
    center = size // 2
    
    # Tạo hình dạng gap dựa trên đặc trưng piece
    if piece_info['circularity'] > 0.6:  # Nếu piece tròn
        # Gap hình tròn
        radius = size // 3
        y, x = np.ogrid[:size, :size]
        mask = (x - center)**2 + (y - center)**2 <= radius**2
        template[mask] = 80
    else:
        # Gap hình vuông/chữ nhật
        w_gap = size // 3
        h_gap = int(w_gap / piece_info['aspect_ratio'])
        
        top = center - h_gap // 2
        bottom = center + h_gap // 2
        left = center - w_gap // 2
        right = center + w_gap // 2
        
        template[top:bottom, left:right] = 80
    
    # Thêm indentations dựa trên solidity
    if piece_info['solidity'] < 0.8:  # Piece có nhiều indentation
        indent_size = max(2, size // 12)
        # Top indent
        template[0:indent_size, center-indent_size:center+indent_size] = 100
        # Bottom indent
        template[size-indent_size:size, center-indent_size:center+indent_size] = 100
        # Left indent
        template[center-indent_size:center+indent_size, 0:indent_size] = 100
        # Right indent
        template[center-indent_size:center+indent_size, size-indent_size:size] = 100
    
    # Smooth template
    template = filters.gaussian(template, sigma=1.0)
    
    return template

def advanced_nms(candidates, distance_threshold=20, score_threshold=0):
    """
    Advanced Non-Maximum Suppression
    """
    if not candidates:
        return []
    
    # Filter by score threshold
    candidates = [c for c in candidates if c['score'] >= score_threshold]
    
    # Sort by score descending
    candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
    
    filtered = []
    for candidate in candidates:
        x, y = candidate['x'], candidate['y']
        
        # Check distance to existing candidates
        too_close = False
        for existing in filtered:
            distance = np.sqrt((x - existing['x'])**2 + (y - existing['y'])**2)
            if distance < distance_threshold:
                too_close = True
                break
        
        if not too_close:
            filtered.append(candidate)
    
    return filtered

def ensemble_scoring(candidates):
    """
    Ensemble scoring để cải thiện độ chính xác
    """
    if len(candidates) <= 1:
        return candidates
    
    # Group candidates by proximity
    groups = []
    for candidate in candidates:
        added_to_group = False
        for group in groups:
            # Check if candidate is close to any member of the group
            for member in group:
                distance = np.sqrt((candidate['x'] - member['x'])**2 + 
                                 (candidate['y'] - member['y'])**2)
                if distance < 40:  # Group threshold
                    group.append(candidate)
                    added_to_group = True
                    break
            if added_to_group:
                break
        
        if not added_to_group:
            groups.append([candidate])
    
    # Create ensemble candidates
    ensemble_candidates = []
    for group in groups:
        if len(group) == 1:
            ensemble_candidates.append(group[0])
        else:
            # Weighted average
            total_weight = sum(c['score'] for c in group)
            if total_weight > 0:
                ensemble_x = sum(c['x'] * c['score'] for c in group) / total_weight
                ensemble_y = sum(c['y'] * c['score'] for c in group) / total_weight
                ensemble_score = sum(c['score'] for c in group) / len(group) * 1.2  # Bonus for consensus
                
                ensemble_candidates.append({
                    'x': int(ensemble_x),
                    'y': int(ensemble_y),
                    'score': ensemble_score,
                    'area': sum(c['area'] for c in group) / len(group),
                    'method': f'ensemble_{len(group)}',
                    'group_size': len(group)
                })
    
    # Sort by score
    ensemble_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return ensemble_candidates

def find_puzzle_gap_ultimate_precision(main_image_path, save_visualization=True):
    """
    Thuật toán tối ưu nhận diện hình ảnh để tìm vị trí ghép mảnh chính xác
    
    Args:
        main_image_path: Đường dẫn đến ảnh captcha
        save_visualization: True = lưu ảnh kết quả, False = không lưu (tiết kiệm bộ nhớ)
    """
    print("🎯 BẮT ĐẦU GIẢI CAPTCHA PUZZLE...")
    
    # Đọc ảnh
    img = cv2.imread(main_image_path)
    if img is None:
        print(f"❌ Không thể đọc ảnh: {main_image_path}")
        return None, None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"📐 Kích thước ảnh: {w}x{h}")
    
    # Phân tích cấu trúc ảnh
    analyze_image_structure(gray)
    
    # Bước 1: Tìm mảnh ghép thông minh
    piece_info = find_puzzle_piece_smart(img, gray)
    if piece_info is None:
        print("❌ Không tìm thấy mảnh ghép!")
        return None, None
    
    piece_template, piece_bbox = piece_info
    print(f"🧩 Đã tìm thấy mảnh ghép: {piece_bbox}")
    
    # Bước 2: Tìm vị trí khớp tốt nhất
    gap_position = find_best_match_position(gray, piece_template, piece_bbox)
    
    if gap_position is None:
        print("❌ Không tìm thấy vị trí khớp!")
        return None, None
    
    # Bước 3: Lưu kết quả visualization (chỉ khi cần)
    if save_visualization:
        save_result_image(img, piece_bbox, gap_position)
        print(f"💾 Đã lưu ảnh visualization")
    else:
        print(f"🚫 Bỏ qua lưu ảnh để tiết kiệm bộ nhớ")
    
    x, y = gap_position
    print(f"✅ THÀNH CÔNG! Vị trí cần click: ({x}, {y})")
    return x, y

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
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/captcha_solved_{timestamp}.png"
    cv2.imwrite(result_path, result_img)
    
    print(f"💾 Đã lưu kết quả: {result_path}")

import base64
import io
import json
import traceback
import tempfile
import os

# Import Flask chỉ khi cần thiết
try:
    from flask import Flask, request, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("⚠️ Flask không có sẵn - chỉ chạy được chế độ solve trực tiếp")

def decode_base64_image(base64_string):
    """
    Decode base64 string thành PIL Image
    """
    try:
        # Loại bỏ header nếu có (data:image/png;base64,)
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Tạo PIL Image từ bytes
        image = Image.open(io.BytesIO(image_data))
        
        return image
    except Exception as e:
        print(f"Lỗi decode base64: {e}")
        return None

def save_temp_image(image, format='PNG'):
    """
    Lưu PIL Image vào file tạm thời
    """
    try:
        # Tạo file tạm thời
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format.lower()}')
        
        # Lưu image
        image.save(temp_file.name, format=format)
        
        return temp_file.name
    except Exception as e:
        print(f"Lỗi lưu file tạm: {e}")
        return None

def process_captcha_image(base64_image):
    """
    Xử lý hình ảnh captcha từ base64 và trả về tọa độ
    """
    try:
        print("🔍 Bắt đầu xử lý captcha...")
        
        # Decode base64 thành image
        image = decode_base64_image(base64_image)
        if image is None:
            return {
                'success': False,
                'error': 'Không thể decode base64 image',
                'coordinates': None
            }
        
        print(f"✅ Decode thành công, kích thước: {image.size}")
        
        # Lưu vào file tạm thời
        temp_path = save_temp_image(image, 'PNG')
        if temp_path is None:
            return {
                'success': False,
                'error': 'Không thể lưu file tạm thời',
                'coordinates': None
            }
        
        print(f"📁 Lưu file tạm: {temp_path}")
        
        try:
            # Xử lý với thuật toán siêu chính xác (không lưu ảnh để tiết kiệm bộ nhớ)
            x, y = find_puzzle_gap_ultimate_precision(temp_path, save_visualization=False)
            
            if x is not None and y is not None:
                # Điều chỉnh tọa độ (nếu cần)
                adjusted_x = x - 18  # Theo code gốc
                adjusted_y = y
                
                result = {
                    'success': True,
                    'coordinates': {
                        'x': adjusted_x,
                        'y': adjusted_y,
                        'raw_x': x,
                        'raw_y': y
                    },
                    'message': 'Tìm thấy vị trí puzzle gap thành công'
                }
                
                print(f"🎯 Thành công: ({adjusted_x}, {adjusted_y})")
                return result
            else:
                return {
                    'success': False,
                    'error': 'Không tìm thấy vị trí puzzle gap',
                    'coordinates': None
                }
                
        finally:
            # Xóa file tạm thời
            try:
                os.unlink(temp_path)
                print(f"🗑️ Đã xóa file tạm: {temp_path}")
            except:
                pass
        
    except Exception as e:
        error_msg = f"Lỗi xử lý: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        return {
            'success': False,
            'error': error_msg,
            'coordinates': None
        }

def create_flask_app():
    """
    Tạo Flask app với API endpoints
    """
    if not FLASK_AVAILABLE:
        print("❌ Flask không có sẵn! Không thể tạo API server.")
        return None
    
    app = Flask(__name__)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'Captcha Puzzle Solver',
            'version': '1.0.0'
        })
    
    @app.route('/solve-captcha', methods=['POST'])
    def solve_captcha():
        """
        API endpoint để xử lý captcha
        Body: {
            "image": "base64_string",
            "format": "png" (optional)
        }
        """
        try:
            # Kiểm tra content type
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Content-Type phải là application/json'
                }), 400
            
            # Lấy data từ request
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Thiếu trường "image" trong request body'
                }), 400
            
            base64_image = data['image']
            
            if not base64_image:
                return jsonify({
                    'success': False,
                    'error': 'Base64 image không được để trống'
                }), 400
            
            # Xử lý captcha
            result = process_captcha_image(base64_image)
            
            if result['success']:
                return jsonify(result), 200
            else:
                return jsonify(result), 422  # Unprocessable Entity
            
        except Exception as e:
            error_msg = f"Lỗi server: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            
            return jsonify({
                'success': False,
                'error': error_msg,
                'coordinates': None
            }), 500
    
    @app.route('/solve-captcha-batch', methods=['POST'])
    def solve_captcha_batch():
        """
        API endpoint để xử lý nhiều captcha cùng lúc
        Body: {
            "images": ["base64_string1", "base64_string2", ...],
            "format": "png" (optional)
        }
        """
        try:
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Content-Type phải là application/json'
                }), 400
            
            data = request.get_json()
            
            if not data or 'images' not in data:
                return jsonify({
                    'success': False,
                    'error': 'Thiếu trường "images" trong request body'
                }), 400
            
            images = data['images']
            
            if not isinstance(images, list) or len(images) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Trường "images" phải là array không rỗng'
                }), 400
            
            # Giới hạn số lượng ảnh để tránh quá tải
            if len(images) > 10:
                return jsonify({
                    'success': False,
                    'error': 'Tối đa 10 ảnh mỗi lần xử lý'
                }), 400
            
            # Xử lý từng ảnh
            results = []
            for i, base64_image in enumerate(images):
                print(f"🔄 Xử lý ảnh {i+1}/{len(images)}")
                
                result = process_captcha_image(base64_image)
                result['index'] = i
                results.append(result)
            
            # Thống kê kết quả
            success_count = sum(1 for r in results if r['success'])
            
            return jsonify({
                'success': True,
                'total': len(images),
                'success_count': success_count,
                'failed_count': len(images) - success_count,
                'results': results
            }), 200
            
        except Exception as e:
            error_msg = f"Lỗi server: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
    
    return app

def run_api_server(host='0.0.0.0', port=8080, debug=False):
    """
    Chạy API server
    """
    app = create_flask_app()
    
    print("🚀 Khởi động Captcha Puzzle Solver API...")
    print(f"📡 Server: http://{host}:{port}")
    print("📋 Endpoints:")
    print("  GET  /health - Health check")
    print("  POST /solve-captcha - Xử lý 1 captcha")
    print("  POST /solve-captcha-batch - Xử lý nhiều captcha")
    print("\n📖 Ví dụ sử dụng:")
    print("""
curl -X POST http://localhost:8080/solve-captcha \\
  -H "Content-Type: application/json" \\
  -d '{
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
  }'
    """)
    
    # Use Gunicorn for production
    if os.environ.get('FLASK_ENV') == 'production':
        import gunicorn.app.base
        
        class StandaloneApplication(gunicorn.app.base.BaseApplication):
            def __init__(self, app, options=None):
                self.options = options or {}
                self.application = app
                super().__init__()
            
            def load_config(self):
                for key, value in self.options.items():
                    self.cfg.set(key.lower(), value)
            
            def load(self):
                return self.application
        
        options = {
            'bind': f'{host}:{port}',
            'workers': 2,
            'worker_class': 'sync',
            'worker_connections': 1000,
            'timeout': 300,
            'keepalive': 5,
            'max_requests': 1000,
            'max_requests_jitter': 100,
            'loglevel': 'info'
        }
        
        StandaloneApplication(app, options).run()
    else:
        app.run(host=host, port=port, debug=debug)

# Hàm helper để test
def test_with_local_image(image_path):
    """
    Test function với ảnh local
    """
    try:
        # Đọc ảnh và convert sang base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
        
        print(f"📷 Test với ảnh: {image_path}")
        print(f"📦 Base64 length: {len(base64_string)}")
        
        # Xử lý
        result = process_captcha_image(base64_string)
        
        print("🔍 Kết quả:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        return result
        
    except Exception as e:
        print(f"❌ Lỗi test: {e}")
        return None

# Sử dụng
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # Chạy API server với cấu hình mặc định
        run_api_server(host='0.0.0.0', port=8080, debug=False)
    
    elif sys.argv[1] == "api":
        # Chạy API server
        host = sys.argv[2] if len(sys.argv) > 2 else '0.0.0.0'
        port = int(sys.argv[3]) if len(sys.argv) > 3 else 8080
        debug = len(sys.argv) > 4 and sys.argv[4].lower() == 'debug'
        
        run_api_server(host=host, port=port, debug=debug)
    
    elif sys.argv[1] == "test" and len(sys.argv) > 2:
        # Test với ảnh local
        image_path = sys.argv[2]
        test_with_local_image(image_path)
    
    elif sys.argv[1] == "solve" and len(sys.argv) > 2:
        # Xử lý trực tiếp
        main_image = sys.argv[2]
        
        if not os.path.exists(main_image):
            print(f"❌ Không tìm thấy file: {main_image}")
            sys.exit(1)
        
        print("🔍 THUẬT TOÁN SIÊU CHÍNH XÁC")
        print(f"📷 Xử lý ảnh: {main_image}")
        
        x, y = find_puzzle_gap_ultimate_precision(main_image)
        
        if x is not None and y is not None:
            print(f"\n🏆 HOÀN THÀNH SIÊU CHÍNH XÁC!")
            print(f"Kết quả cuối cùng: ({x-18}, {y})")
        else:
            print("❌ Thất bại")
    else:
        print("Cách sử dụng:")
        print("  python resolveCaptcha.py                    # Chạy API server (port 8080)")
        print("  python resolveCaptcha.py api [host] [port]  # Chạy API server")
        print("  python resolveCaptcha.py test image.png     # Test với ảnh local")
        print("  python resolveCaptcha.py solve image.png    # Xử lý trực tiếp")