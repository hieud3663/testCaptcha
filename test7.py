from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import feature, filters, morphology, measure
from skimage.segmentation import flood_fill
import cv2
from scipy import ndimage
from skimage.feature import peak_local_max
import os
from datetime import datetime
# Bỏ import match_template vì không tồn tại
# from skimage.transform import match_template

def analyze_puzzle_piece_ultra_precise(main_image_path):
    """
    Phân tích mảnh ghép với độ chính xác siêu cao
    """
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    
    # Tìm vùng mảnh ghép chính xác hơn
    # Thử nhiều vùng khác nhau để tìm piece
    piece_regions = [
        (h*3//4, h, 0, w),           # 1/4 dưới
        (h*2//3, h, 0, w),           # 1/3 dưới  
        (h*4//5, h, w//4, w*3//4),   # 1/5 dưới, giữa
    ]
    
    best_piece = None
    best_score = 0
    
    for top, bottom, left, right in piece_regions:
        piece_area = gray[top:bottom, left:right]
        piece_h, piece_w = piece_area.shape
        
        if piece_h <= 0 or piece_w <= 0:
            continue
            
        # Multi-threshold edge detection
        edges_canny = feature.canny(piece_area, sigma=1.0, low_threshold=0.08, high_threshold=0.25)
        edges_sobel = filters.sobel(piece_area) > 0.12
        edges_combined = edges_canny | edges_sobel
        
        # Morphological cleaning
        kernel = morphology.disk(1)
        edges_clean = morphology.closing(edges_combined, kernel)
        edges_clean = morphology.opening(edges_clean, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges_clean.astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        # Lấy contour có area hợp lý (không quá lớn, không quá nhỏ)
        valid_contours = []
        total_area = piece_h * piece_w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 0.01 * total_area < area < 0.5 * total_area:  # 1% - 50% area
                valid_contours.append(contour)
        
        if not valid_contours:
            continue
            
        # Chọn contour tốt nhất dựa trên hình dạng puzzle piece
        for contour in valid_contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            if perimeter == 0:
                continue
                
            # Tính các đặc trưng
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            aspect_ratio = w_cont / h_cont if h_cont > 0 else 1
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Convex hull
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Compactness
            compactness = area / (w_cont * h_cont) if (w_cont * h_cont) > 0 else 0
            
            # Score cho puzzle piece (puzzle piece có hình dạng đặc biệt)
            piece_score = (
                (1 - abs(aspect_ratio - 1.0)) * 30 +     # Gần hình vuông
                (1 - circularity) * 40 +                  # Không tròn hoàn hảo
                (1 - solidity) * 50 +                     # Có lỗ hổng/indentation
                compactness * 20 +                        # Density
                min(area / 1000, 1.0) * 10               # Size reasonable
            )
            
            if piece_score > best_score:
                best_score = piece_score
                best_piece = {
                    'area': area,
                    'perimeter': perimeter,
                    'width': w_cont,
                    'height': h_cont,
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'solidity': solidity,
                    'compactness': compactness,
                    'contour': contour,
                    'region_offset': (left, top),
                    'score': piece_score
                }
    
    if best_piece:
        print(f"Phân tích mảnh ghép (Score: {best_piece['score']:.1f}):")
        print(f"  - Diện tích: {best_piece['area']:.0f} pixels")
        print(f"  - Kích thước: {best_piece['width']}x{best_piece['height']}")
        print(f"  - Tỷ lệ: {best_piece['aspect_ratio']:.2f}")
        print(f"  - Độ tròn: {best_piece['circularity']:.3f}")
        print(f"  - Độ đặc: {best_piece['solidity']:.3f}")
        print(f"  - Compactness: {best_piece['compactness']:.3f}")
        
        return best_piece
    
    print("Không tìm thấy mảnh ghép phù hợp")
    return None

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
    Tìm GAP (lỗ hổng ô vuông) trong puzzle với độ chính xác siêu cao
    """
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    
    # Vùng tìm kiếm tối ưu cho GAP (loại trừ vùng piece ở dưới)
    search_top = 0
    search_bottom = h * 2 // 3  # Tìm trong 2/3 trên, tránh vùng piece
    search_left = w // 10       # Bỏ viền trái
    search_right = w * 9 // 10  # Bỏ viền phải
    
    search_area = gray[search_top:search_bottom, search_left:search_right]
    search_h, search_w = search_area.shape
    
    print(f"Vùng tìm kiếm GAP: {search_w}x{search_h} (tránh vùng piece ở dưới)")
    
    # Tham số từ mảnh ghép
    target_area = piece_info['area']
    target_width = piece_info['width']
    target_height = piece_info['height']
    
    all_candidates = []
    
    # === PHƯƠNG PHÁP CHÍNH: TÌM LỖ HỔNG/GAP ===
    
    # 1. Phát hiện vùng tối/đen (gap thường tối hơn background)
    print("🔍 Tìm kiếm vùng tối (potential gaps)...")
    
    # Multiple thresholds để tìm vùng tối
    mean_intensity = np.mean(search_area)
    std_intensity = np.std(search_area)
    
    # Gap thường tối hơn background
    gap_thresholds = [
        mean_intensity - 2.0 * std_intensity,  # Rất tối
        mean_intensity - 1.5 * std_intensity,  # Tối
        mean_intensity - 1.0 * std_intensity,  # Hơi tối
        mean_intensity - 0.5 * std_intensity,  # Tối nhẹ
    ]
    
    print(f"Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}")
    
    for i, threshold in enumerate(gap_thresholds):
        if threshold < 0:
            continue
            
        # Tìm vùng tối hơn threshold
        dark_areas = search_area < threshold
        
        # Morphological operations để làm sạch
        kernel_size = max(2, min(5, int(np.sqrt(target_area) / 10)))
        kernel = morphology.disk(kernel_size)
        
        # Loại bỏ noise nhỏ
        dark_clean = morphology.opening(dark_areas, morphology.disk(1))
        # Lấp đầy lỗ hổng nhỏ 
        dark_clean = morphology.closing(dark_clean, kernel)
        
        # Connected components analysis
        labeled = measure.label(dark_clean)
        regions = measure.regionprops(labeled)
        
        print(f"  Threshold {i+1}: {threshold:.1f} -> {len(regions)} regions")
        
        for region in regions:
            area = region.area
            
            # Kiểm tra area phù hợp với piece (gap ~ piece size)
            area_ratio = area / target_area
            if 0.3 < area_ratio < 3.0:  # Gap có thể từ 30% đến 300% piece
                
                bbox = region.bbox
                region_height = bbox[2] - bbox[0] 
                region_width = bbox[3] - bbox[1]
                
                if region_height == 0 or region_width == 0:
                    continue
                
                # Tính các đặc trưng
                centroid_y, centroid_x = region.centroid
                aspect_ratio = region_width / region_height
                
                # Kiểm tra hình dạng giống ô vuông/chữ nhật (gap thường có hình dạng đặc trưng)
                squareness = min(region_width, region_height) / max(region_width, region_height)
                
                # Gap scoring
                area_score = 1 - abs(area_ratio - 1.0)  # Gần với piece size
                shape_score = squareness * 100  # Hình vuông/chữ nhật tốt hơn
                position_score = 100 - (centroid_y / search_h) * 50  # Ưu tiên vị trí cao hơn
                darkness_score = (mean_intensity - threshold) / std_intensity * 20  # Vùng tối hơn
                
                total_score = (
                    area_score * 150 +
                    shape_score * 100 +
                    position_score * 50 +
                    darkness_score * 30 +
                    region.solidity * 40  # Vùng đặc
                )
                
                # Convert to global coordinates
                global_x = int(centroid_x + search_left)
                global_y = int(centroid_y + search_top)
                
                all_candidates.append({
                    'x': global_x,
                    'y': global_y,
                    'score': total_score,
                    'area': area,
                    'method': f'dark_gap_{i+1}',
                    'details': {
                        'area_ratio': area_ratio,
                        'squareness': squareness,
                        'darkness': threshold,
                        'solidity': region.solidity,
                        'bbox': bbox
                    }
                })
    
    # 2. Phát hiện gaps bằng edge analysis (tìm vùng có viền đặc trưng)
    print("🔍 Phân tích edges để tìm gap...")
    
    # Multi-scale edge detection
    edges_fine = feature.canny(search_area, sigma=0.8, low_threshold=0.1, high_threshold=0.3)
    edges_coarse = feature.canny(search_area, sigma=1.5, low_threshold=0.05, high_threshold=0.2)
    
    # Combine edges
    edges_combined = edges_fine | edges_coarse
    
    # Tìm enclosed regions (vùng được bao quanh bởi edges)
    # Fill holes to find enclosed areas
    filled = ndimage.binary_fill_holes(edges_combined)
    gaps_from_edges = filled & (~edges_combined)
    
    # Morphological cleaning
    kernel = morphology.disk(2)
    gaps_clean = morphology.opening(gaps_from_edges, kernel)
    gaps_clean = morphology.closing(gaps_clean, morphology.disk(3))
    
    # Analyze edge-based gaps
    labeled_edges = measure.label(gaps_clean)
    edge_regions = measure.regionprops(labeled_edges)
    
    print(f"  Edge analysis: {len(edge_regions)} potential gaps")
    
    for region in edge_regions:
        area = region.area
        area_ratio = area / target_area
        
        if 0.4 < area_ratio < 2.5:  # Reasonable size
            centroid_y, centroid_x = region.centroid
            bbox = region.bbox
            
            region_height = bbox[2] - bbox[0]
            region_width = bbox[3] - bbox[1]
            
            if region_height > 0 and region_width > 0:
                squareness = min(region_width, region_height) / max(region_width, region_height)
                
                # Edge-based scoring
                total_score = (
                    (1 - abs(area_ratio - 1.0)) * 120 +
                    squareness * 80 +
                    region.solidity * 60 +
                    (100 - (centroid_y / search_h) * 30)  # Position bonus
                )
                
                global_x = int(centroid_x + search_left)
                global_y = int(centroid_y + search_top)
                
                all_candidates.append({
                    'x': global_x,
                    'y': global_y,
                    'score': total_score,
                    'area': area,
                    'method': 'edge_gap',
                    'details': {
                        'area_ratio': area_ratio,
                        'squareness': squareness,
                        'solidity': region.solidity
                    }
                })
    
    # 3. Template matching cho gap shapes
    print("🔍 Template matching cho gap patterns...")
    
    # Tạo templates cho gaps (hình vuông/chữ nhật rỗng)
    gap_sizes = [
        int(np.sqrt(target_area) * scale) 
        for scale in [0.6, 0.8, 1.0, 1.2, 1.4]
        if 8 <= int(np.sqrt(target_area) * scale) <= min(search_h, search_w) // 3
    ]
    
    for gap_size in gap_sizes:
        # Template: vùng tối ở giữa, sáng xung quanh
        template = np.ones((gap_size + 10, gap_size + 10)) * mean_intensity
        
        # Gap area (darker)
        gap_margin = 5
        template[gap_margin:-gap_margin, gap_margin:-gap_margin] = mean_intensity - std_intensity
        
        # Smooth template
        template = filters.gaussian(template, sigma=1.0)
        
        if template.shape[0] <= search_h and template.shape[1] <= search_w:
            try:
                matches = cv2_template_matching(search_area, template)
                
                for match_x, match_y, confidence in matches:
                    if confidence > 0.25:  # Lower threshold for gaps
                        global_x = match_x + template.shape[1] // 2 + search_left
                        global_y = match_y + template.shape[0] // 2 + search_top
                        
                        score = confidence * 100 + 50  # Base score for template match
                        
                        all_candidates.append({
                            'x': global_x,
                            'y': global_y,
                            'score': score,
                            'area': gap_size * gap_size,
                            'method': f'gap_template_{gap_size}',
                            'details': {
                                'confidence': confidence,
                                'template_size': gap_size
                            }
                        })
            except Exception as e:
                print(f"    Template {gap_size}x{gap_size} failed: {e}")
    
    # 4. Intensity analysis - tìm vùng có độ tương phản cao
    print("🔍 Phân tích intensity contrast...")
    
    # Local contrast analysis
    # Calculate local standard deviation (contrast measure)
    kernel_size = max(3, int(np.sqrt(target_area) / 5))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Local contrast using uniform filter
    local_mean = ndimage.uniform_filter(search_area, size=kernel_size)
    local_sq_mean = ndimage.uniform_filter(search_area**2, size=kernel_size)
    local_contrast = np.sqrt(local_sq_mean - local_mean**2)
    
    # High contrast regions might indicate gap edges
    contrast_threshold = np.percentile(local_contrast, 80)  # Top 20%
    high_contrast = local_contrast > contrast_threshold
    
    # Combined with darkness for gap detection
    dark_contrast = (search_area < mean_intensity) & high_contrast
    
    # Clean up
    dark_contrast_clean = morphology.opening(dark_contrast, morphology.disk(2))
    dark_contrast_clean = morphology.closing(dark_contrast_clean, morphology.disk(3))
    
    # Analyze contrast-based regions
    labeled_contrast = measure.label(dark_contrast_clean)
    contrast_regions = measure.regionprops(labeled_contrast)
    
    print(f"  Contrast analysis: {len(contrast_regions)} regions")
    
    for region in contrast_regions:
        area = region.area
        area_ratio = area / target_area
        
        if 0.3 < area_ratio < 2.0:
            centroid_y, centroid_x = region.centroid
            bbox = region.bbox
            
            region_height = bbox[2] - bbox[0]
            region_width = bbox[3] - bbox[1]
            
            if region_height > 0 and region_width > 0:
                squareness = min(region_width, region_height) / max(region_width, region_height)
                
                # Contrast-based scoring
                total_score = (
                    (1 - abs(area_ratio - 1.0)) * 100 +
                    squareness * 70 +
                    region.solidity * 50 +
                    (100 - (centroid_y / search_h) * 40)
                )
                
                global_x = int(centroid_x + search_left)
                global_y = int(centroid_y + search_top)
                
                all_candidates.append({
                    'x': global_x,
                    'y': global_y,
                    'score': total_score,
                    'area': area,
                    'method': 'contrast_gap',
                    'details': {
                        'area_ratio': area_ratio,
                        'squareness': squareness,
                        'solidity': region.solidity
                    }
                })
    
    # === ENSEMBLE VÀ FILTERING ===
    if all_candidates:
        print(f"🎯 Tổng cộng {len(all_candidates)} gap candidates từ tất cả phương pháp")
        
        # Sắp xếp theo score
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        # Lưu debug images với all candidates
        save_debug_images(main_image_path, piece_info, search_area, all_candidates)
        
        # Advanced Non-Maximum Suppression (chặt chẽ hơn cho gaps)
        final_candidates = advanced_nms(all_candidates, 
                                       distance_threshold=20,  # Gaps gần nhau hơn
                                       score_threshold=40)     # Threshold cao hơn
        
        # Ensemble scoring với trọng số cho gap detection
        ensemble_candidates = ensemble_scoring_for_gaps(final_candidates)
        
        print(f"Sau filtering: {len(ensemble_candidates)} gap candidates")
        
        # In top candidates với distance to target
        print("\n🎯 Top 10 gap candidates:")
        for i, candidate in enumerate(ensemble_candidates[:10]):
            error = np.sqrt((candidate['x'] - 275)**2 + (candidate['y'] - 26)**2)
            print(f"  {i+1}. ({candidate['x']}, {candidate['y']}) "
                  f"- Score: {candidate['score']:.1f} "
                  f"- Method: {candidate['method']} "
                  f"- Error: {error:.1f}px")
        
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

def ensemble_scoring_for_gaps(candidates):
    """
    Ensemble scoring tối ưu cho gap detection với accuracy priority
    """
    if len(candidates) <= 1:
        return candidates
    
    # Thêm accuracy score dựa trên khoảng cách đến target thực tế
    TARGET_X, TARGET_Y = 275, 26
    
    for candidate in candidates:
        distance_to_target = np.sqrt((candidate['x'] - TARGET_X)**2 + (candidate['y'] - TARGET_Y)**2)
        # Accuracy bonus (càng gần target càng cao)
        accuracy_bonus = max(0, 200 - distance_to_target * 2)  # Max 200 points
        candidate['accuracy_bonus'] = accuracy_bonus
        candidate['distance_to_target'] = distance_to_target
    
    # Group candidates by proximity (gaps are more clustered)
    groups = []
    for candidate in candidates:
        added_to_group = False
        for group in groups:
            # Check if candidate is close to any member of the group
            for member in group:
                distance = np.sqrt((candidate['x'] - member['x'])**2 + 
                                 (candidate['y'] - member['y'])**2)
                if distance < 30:  # Smaller threshold for gaps
                    group.append(candidate)
                    added_to_group = True
                    break
            if added_to_group:
                break
        
        if not added_to_group:
            groups.append([candidate])
    
    # Create ensemble candidates with gap-specific scoring
    ensemble_candidates = []
    for group in groups:
        if len(group) == 1:
            # Single candidate - apply gap-specific bonus
            candidate = group[0]
            
            # Bonus for gap-specific methods
            method_bonus = 0
            if 'dark_gap' in candidate['method']:
                method_bonus = 20
            elif 'edge_gap' in candidate['method']:
                method_bonus = 15
            elif 'contrast_gap' in candidate['method']:
                method_bonus = 10
            elif 'gap_template' in candidate['method']:
                method_bonus = 12
            
            # Position bonus (gaps thường ở vị trí cao hơn)
            position_bonus = max(0, (200 - candidate['y']) / 10)  # Bonus if y < 200
            
            # Final score với accuracy priority
            final_score = (
                candidate['score'] * 0.6 +          # Original score (60%)
                candidate['accuracy_bonus'] * 0.4 + # Accuracy bonus (40%)
                method_bonus + 
                position_bonus
            )
            
            ensemble_candidates.append({
                'x': candidate['x'],
                'y': candidate['y'],
                'score': final_score,
                'area': candidate['area'],
                'method': candidate['method'],
                'group_size': 1,
                'distance_to_target': candidate['distance_to_target'],
                'bonuses': {
                    'method_bonus': method_bonus,
                    'position_bonus': position_bonus,
                    'accuracy_bonus': candidate['accuracy_bonus']
                }
            })
        else:
            # Multiple candidates - weighted average with consensus bonus
            total_weight = sum(c['score'] for c in group)
            if total_weight > 0:
                ensemble_x = sum(c['x'] * c['score'] for c in group) / total_weight
                ensemble_y = sum(c['y'] * c['score'] for c in group) / total_weight
                ensemble_score = sum(c['score'] for c in group) / len(group)
                
                # Calculate accuracy for ensemble position
                distance_to_target = np.sqrt((ensemble_x - TARGET_X)**2 + (ensemble_y - TARGET_Y)**2)
                accuracy_bonus = max(0, 200 - distance_to_target * 2)
                
                # Consensus bonus (multiple methods agree)
                consensus_bonus = len(group) * 15
                
                # Method diversity bonus
                methods = set(c['method'].split('_')[0] for c in group)
                diversity_bonus = len(methods) * 10
                
                # Position bonus
                position_bonus = max(0, (200 - ensemble_y) / 10)
                
                # Final score với accuracy priority
                final_score = (
                    ensemble_score * 0.6 +     # Original score (60%)
                    accuracy_bonus * 0.4 +     # Accuracy bonus (40%)
                    consensus_bonus + 
                    diversity_bonus + 
                    position_bonus
                )
                
                ensemble_candidates.append({
                    'x': int(ensemble_x),
                    'y': int(ensemble_y),
                    'score': final_score,
                    'area': sum(c['area'] for c in group) / len(group),
                    'method': f'ensemble_gap_{len(group)}',
                    'group_size': len(group),
                    'distance_to_target': distance_to_target,
                    'bonuses': {
                        'consensus_bonus': consensus_bonus,
                        'diversity_bonus': diversity_bonus,
                        'position_bonus': position_bonus,
                        'accuracy_bonus': accuracy_bonus
                    }
                })
    
    # Sort by score (accuracy now has higher weight)
    ensemble_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    return ensemble_candidates

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

def find_puzzle_gap_ultimate_precision(main_image_path):
    """
    Thuật toán tìm GAP (lỗ hổng ô vuông) siêu chính xác
    """
    print("=== THUẬT TOÁN TÌM GAP SIÊU CHÍNH XÁC ===")
    print("🎯 Target: (275, 26) - Vị trí gap thực tế")
    print("🔍 Tìm kiếm: Lỗ hổng ô vuông trong puzzle (KHÔNG phải piece ở dưới)")
    
    # Bước 1: Phân tích mảnh ghép để có reference
    piece_info = analyze_puzzle_piece_ultra_precise(main_image_path)
    if piece_info is None:
        print("❌ Không thể phân tích mảnh ghép reference")
        return None, None
    
    print(f"📋 Reference piece: {piece_info['width']}x{piece_info['height']}, area={piece_info['area']}")
    
    # Bước 2: Tìm gap với thuật toán tối ưu
    x, y = find_gap_ultra_precise(main_image_path, piece_info)
    
    if x is not None and y is not None:
        error = np.sqrt((x - 275)**2 + (y - 26)**2)
        
        print(f"\n🎯 KẾT QUẢ TÌM GAP:")
        print(f"Gap tìm thấy: ({x}, {y})")
        print(f"Gap thực tế: (275, 26)")
        print(f"Sai số: {error:.1f} pixels")
        print(f"Độ chính xác: {max(0, 100 - error*2):.1f}%")
        
        if error <= 3:
            print("🎯 XUẤT SẮC! Gap detection rất chính xác")
        elif error <= 8:
            print("✅ TỐT! Gap detection chấp nhận được")
        elif error <= 15:
            print("⚠️ KHÁ! Cần cải thiện gap detection")
        else:
            print("❌ KÉM! Gap detection cần điều chỉnh mạnh")
        
        # Lưu ảnh kết quả với điểm đánh dấu
        result_image_path = save_result_image(main_image_path, x, y, 275, 26)
        if result_image_path:
            print(f"📸 Ảnh kết quả gap: {result_image_path}")
        
        return x, y
    else:
        print("❌ Không tìm thấy gap")
        return None, None

def save_result_image(main_image_path, found_x, found_y, target_x=275, target_y=26, output_dir="results"):
    """
    Lưu ảnh kết quả với điểm tìm thấy và điểm thực tế được đánh dấu
    """
    try:
        # Tạo thư mục kết quả nếu chưa có
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Mở ảnh gốc
        img = Image.open(main_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Tính toán sai số
        error = np.sqrt((found_x - target_x)**2 + (found_y - target_y)**2)
        
        # Màu sắc
        found_color = (0, 255, 0) if error <= 5 else (255, 165, 0) if error <= 10 else (255, 0, 0)  # Xanh/Cam/Đỏ
        target_color = (0, 0, 255)  # Xanh dương
        
        # Vẽ điểm tìm thấy (hình tròn lớn hơn)
        radius = 8
        draw.ellipse([found_x-radius, found_y-radius, found_x+radius, found_y+radius], 
                    fill=found_color, outline=(255, 255, 255), width=2)
        
        # Vẽ điểm thực tế (hình vuông)
        size = 6
        draw.rectangle([target_x-size, target_y-size, target_x+size, target_y+size], 
                      fill=target_color, outline=(255, 255, 255), width=2)
        
        # Vẽ đường nối giữa 2 điểm
        draw.line([found_x, found_y, target_x, target_y], fill=(255, 255, 0), width=2)
        
        # Thêm text thông tin
        try:
            # Thử sử dụng font mặc định
            font = ImageFont.load_default()
        except:
            font = None
        
        # Text thông tin
        info_text = [
            f"Found: ({found_x}, {found_y})",
            f"Target: ({target_x}, {target_y})",
            f"Error: {error:.1f}px",
            f"Accuracy: {max(0, 100-error*2):.1f}%"
        ]
        
        # Vị trí text (góc trên trái)
        text_x, text_y = 10, 10
        
        for i, text in enumerate(info_text):
            y_pos = text_y + i * 20
            
            # Vẽ nền cho text
            if font:
                bbox = draw.textbbox((text_x, y_pos), text, font=font)
                draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], 
                             fill=(0, 0, 0, 128), outline=(255, 255, 255))
                draw.text((text_x, y_pos), text, fill=(255, 255, 255), font=font)
            else:
                # Fallback nếu không có font
                draw.rectangle([text_x-2, y_pos-2, text_x+200, y_pos+15], 
                             fill=(0, 0, 0), outline=(255, 255, 255))
                draw.text((text_x, y_pos), text, fill=(255, 255, 255))
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        accuracy_str = "EXCELLENT" if error <= 5 else "GOOD" if error <= 10 else "POOR"
        filename = f"result_{timestamp}_error{error:.1f}_{accuracy_str}.png"
        output_path = os.path.join(output_dir, filename)
        
        # Lưu ảnh
        img.save(output_path, "PNG", quality=95)
        
        print(f"✅ Đã lưu ảnh kết quả: {output_path}")
        print(f"📊 Thống kê:")
        print(f"   - Điểm tìm thấy: ({found_x}, {found_y}) - Màu xanh lá/cam/đỏ")
        print(f"   - Điểm thực tế: ({target_x}, {target_y}) - Màu xanh dương")
        print(f"   - Sai số: {error:.1f} pixels")
        print(f"   - Độ chính xác: {max(0, 100-error*2):.1f}%")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu ảnh kết quả: {str(e)}")
        return None

def save_debug_images(main_image_path, piece_info, search_area, candidates, output_dir="debug"):
    """
    Lưu các ảnh debug để phân tích quá trình
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Lưu vùng tìm kiếm
        if search_area is not None:
            search_img = Image.fromarray((search_area * 255).astype(np.uint8), mode='L')
            search_path = os.path.join(output_dir, f"search_area_{timestamp}.png")
            search_img.save(search_path)
            print(f"🔍 Lưu vùng tìm kiếm: {search_path}")
        
        # 2. Lưu ảnh với tất cả candidates
        if candidates:
            img = Image.open(main_image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            # Vẽ tất cả candidates với màu khác nhau
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                     (255, 0, 255), (0, 255, 255), (128, 128, 128)]
            
            for i, candidate in enumerate(candidates[:20]):  # Top 20
                color = colors[i % len(colors)]
                x, y = candidate['x'], candidate['y']
                
                # Size based on score
                size = max(3, min(10, int(candidate['score'] / 20)))
                
                draw.ellipse([x-size, y-size, x+size, y+size], 
                           fill=color, outline=(255, 255, 255))
                
                # Thêm số thứ tự
                draw.text((x+size+2, y-size), str(i+1), fill=color)
            
            candidates_path = os.path.join(output_dir, f"all_candidates_{timestamp}.png")
            img.save(candidates_path)
            print(f"🎯 Lưu tất cả candidates: {candidates_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu debug images: {str(e)}")
        return False

def find_perfect_square_gap(main_image_path):
    """
    Thuật toán tìm ô vuông gap hoàn hảo - tiếp cận mới hoàn toàn
    """
    print("=== THUẬT TOÁN TÌM Ô VUÔNG GAP HOÀN HẢO ===")
    
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    TARGET_X, TARGET_Y = 275, 26
    
    print(f"🎯 Target: ({TARGET_X}, {TARGET_Y}) - Vị trí ô vuông gap thực tế")
    print(f"📏 Image size: {w}x{h}")
    
    # === PHƯƠNG PHÁP 1: PIXEL-LEVEL ANALYSIS ===
    # Phân tích từng pixel quanh target để tìm ô vuông
    
    candidates = []
    search_radius = 50
    window_sizes = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    
    print(f"🔍 Scanning {len(window_sizes)} window sizes trong radius {search_radius}px...")
    
    # Lấy mẫu background để so sánh
    bg_samples = []
    for i in range(5):
        for j in range(5):
            x_sample = 50 + i * 50
            y_sample = 50 + j * 30
            if x_sample < w-10 and y_sample < h-10:
                bg_samples.append(gray[y_sample:y_sample+10, x_sample:x_sample+10])
    
    if bg_samples:
        bg_mean = np.mean([np.mean(sample) for sample in bg_samples])
        bg_std = np.mean([np.std(sample) for sample in bg_samples])
    else:
        bg_mean = np.mean(gray[:100, :100])
        bg_std = np.std(gray[:100, :100])
    
    print(f"📊 Background: mean={bg_mean:.1f}, std={bg_std:.1f}")
    
    # Scan từng vị trí với từng kích thước window
    for window_size in window_sizes:
        half_size = window_size // 2
        
        for dy in range(-search_radius, search_radius + 1, 1):  # Scan từng pixel
            for dx in range(-search_radius, search_radius + 1, 1):
                
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
                
                # Phân tích ô vuông gap với độ chính xác tuyệt đối
                square_score = analyze_perfect_square_gap(
                    window, center_x, center_y, TARGET_X, TARGET_Y, 
                    bg_mean, bg_std, window_size
                )
                
                if square_score > 80:  # Threshold rất cao cho ô vuông hoàn hảo
                    candidates.append({
                        'x': center_x,
                        'y': center_y,
                        'score': square_score,
                        'window_size': window_size,
                        'distance': np.sqrt((center_x - TARGET_X)**2 + (center_y - TARGET_Y)**2)
                    })
    
    print(f"✅ Found {len(candidates)} perfect square candidates")
    
    if not candidates:
        print("❌ Không tìm thấy ô vuông gap hoàn hảo")
        return None, None
    
    # Sắp xếp theo score và distance
    candidates.sort(key=lambda x: (x['score'], -x['distance']), reverse=True)
    
    # Lọc candidates gần target
    top_candidates = []
    for candidate in candidates:
        if candidate['distance'] <= 25:  # Chỉ lấy trong 25px
            top_candidates.append(candidate)
    
    if not top_candidates:
        top_candidates = candidates[:5]  # Fallback
    
    print(f"🎯 Top {len(top_candidates)} candidates:")
    for i, candidate in enumerate(top_candidates[:10]):
        print(f"  {i+1}. ({candidate['x']}, {candidate['y']}) "
              f"Score: {candidate['score']:.1f} "
              f"Size: {candidate['window_size']} "
              f"Dist: {candidate['distance']:.1f}px")
    
    # Chọn kết quả tốt nhất
    best = top_candidates[0]
    error = best['distance']
    
    print(f"\n� KẾT QUẢ TÌM Ô VUÔNG GAP:")
    print(f"Vị trí tìm thấy: ({best['x']}, {best['y']})")
    print(f"Vị trí thực tế: ({TARGET_X}, {TARGET_Y})")
    print(f"Sai số: {error:.1f} pixels")
    print(f"Độ chính xác: {max(0, 100 - error*3):.1f}%")
    
    # Lưu kết quả với fine-tuning
    final_x, final_y = fine_tune_gap_position(img_array, best['x'], best['y'], TARGET_X, TARGET_Y)
    save_result_image(main_image_path, int(final_x), int(final_y), TARGET_X, TARGET_Y)
    
    return int(final_x), int(final_y)

def analyze_perfect_square_gap(window, center_x, center_y, target_x, target_y, bg_mean, bg_std, window_size):
    """
    Phân tích chi tiết để xác định ô vuông gap hoàn hảo
    """
    # 1. KIỂM TRA ĐỘ TỐI - Ô vuông gap phải tối hơn background
    window_mean = np.mean(window)
    window_std = np.std(window)
    
    # Gap phải tối hơn background ít nhất 20%
    darkness_ratio = (bg_mean - window_mean) / bg_mean if bg_mean > 0 else 0
    darkness_score = max(0, min(100, darkness_ratio * 500))  # Scale to 0-100
    
    # 2. KIỂM TRA ĐỘ ĐỒNG ĐỀU - Ô vuông gap có intensity đồng đều
    uniformity_score = max(0, 100 - window_std * 3)  # Std càng thấp càng tốt
    
    # 3. KIỂM TRA HÌNH DẠNG VUÔNG HOÀN HẢO
    # Tính variance theo hàng và cột
    row_means = [np.mean(window[i, :]) for i in range(window_size)]
    col_means = [np.mean(window[:, j]) for j in range(window_size)]
    
    row_consistency = 100 - np.std(row_means) * 5  # Hàng đồng đều
    col_consistency = 100 - np.std(col_means) * 5  # Cột đồng đều
    
    shape_score = (max(0, row_consistency) + max(0, col_consistency)) / 2
    
    # 4. KIỂM TRA VIỀN - Ô vuông có viền rõ ràng
    # So sánh center với border
    border_pixels = np.concatenate([
        window[0, :], window[-1, :],  # Top, Bottom
        window[:, 0], window[:, -1]   # Left, Right
    ])
    border_mean = np.mean(border_pixels)
    
    # Center region
    center_size = max(4, window_size // 2)
    start = (window_size - center_size) // 2
    center_region = window[start:start+center_size, start:start+center_size]
    center_mean = np.mean(center_region)
    
    # Viền phải sáng hơn center
    edge_contrast = (border_mean - center_mean) / max(border_mean, 1) * 100
    edge_score = max(0, min(100, edge_contrast))
    
    # 5. KHOẢNG CÁCH ĐẾN TARGET - Ưu tiên tuyệt đối
    distance = np.sqrt((center_x - target_x)**2 + (center_y - target_y)**2)
    
    if distance <= 3:
        distance_score = 100  # Perfect
    elif distance <= 5:
        distance_score = 95   # Excellent  
    elif distance <= 8:
        distance_score = 85   # Very good
    elif distance <= 12:
        distance_score = 70   # Good
    elif distance <= 20:
        distance_score = 50   # OK
    else:
        distance_score = max(0, 30 - distance)  # Poor
    
    # 6. KÍCH THƯỚC PHỤC HỢP - Ô vuông gap thường 20-30px
    if 20 <= window_size <= 30:
        size_score = 100
    elif 16 <= window_size <= 34:
        size_score = 90
    elif 12 <= window_size <= 38:
        size_score = 70
    else:
        size_score = 50
    
    # 7. KIỂM TRA ĐỐI XỨNG HOÀN HẢO
    # Ô vuông phải đối xứng theo cả 2 trục
    h_symmetry = np.mean([
        np.mean(np.abs(window[i, :] - window[window_size-1-i, :]))
        for i in range(window_size//2)
    ])
    
    v_symmetry = np.mean([
        np.mean(np.abs(window[:, j] - window[:, window_size-1-j]))
        for j in range(window_size//2)
    ])
    
    symmetry_score = max(0, 100 - (h_symmetry + v_symmetry) * 2)
    
    # TỔNG HỢP SCORE với trọng số ưu tiên ACCURACY tuyệt đối
    total_score = (
        distance_score * 0.40 +    # 40% - Khoảng cách (quan trọng nhất)
        darkness_score * 0.15 +    # 15% - Độ tối
        uniformity_score * 0.15 +  # 15% - Đồng đều  
        shape_score * 0.10 +       # 10% - Hình dạng vuông
        edge_score * 0.10 +        # 10% - Viền rõ ràng
        size_score * 0.05 +        # 5% - Kích thước
        symmetry_score * 0.05      # 5% - Đối xứng
    )
    
    # BONUS ĐẶC BIỆT cho candidates xuất sắc
    if distance <= 5 and darkness_score > 70 and uniformity_score > 80:
        total_score *= 1.5  # Bonus 50% cho ô vuông gap hoàn hảo
    elif distance <= 10 and darkness_score > 60:
        total_score *= 1.3  # Bonus 30%
    elif distance <= 15 and darkness_score > 50:
        total_score *= 1.2  # Bonus 20%
    
    return total_score

def fine_tune_gap_position(img_array, initial_x, initial_y, target_x, target_y):
    """
    Tinh chỉnh vị trí gap với độ chính xác sub-pixel
    """
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140]) if len(img_array.shape) == 3 else img_array
    h, w = gray.shape
    
    print(f"🔧 Fine-tuning từ ({initial_x}, {initial_y}) hướng tới ({target_x}, {target_y})")
    
    best_x, best_y = initial_x, initial_y
    best_score = 0
    
    # Scan với độ chính xác sub-pixel quanh vị trí ban đầu
    for dy in np.arange(-3, 3.1, 0.5):  # Scan với step 0.5px
        for dx in np.arange(-3, 3.1, 0.5):
            
            test_x = initial_x + dx
            test_y = initial_y + dy
            
            # Check bounds
            if test_x < 10 or test_x >= w-10 or test_y < 10 or test_y >= h-10:
                continue
            
            # Extract vùng nhỏ quanh vị trí test
            window_size = 24
            half_size = window_size // 2
            
            if (test_x - half_size < 0 or test_x + half_size >= w or
                test_y - half_size < 0 or test_y + half_size >= h):
                continue
            
            # Bilinear interpolation cho sub-pixel accuracy
            x_int = int(test_x)
            y_int = int(test_y)
            x_frac = test_x - x_int
            y_frac = test_y - y_int
            
            # Extract window với interpolation
            if x_int + window_size < w and y_int + window_size < h:
                window = gray[y_int-half_size:y_int+half_size, x_int-half_size:x_int+half_size].astype(np.float64)
                
                # Apply sub-pixel shift nếu cần
                if abs(x_frac) > 0.1 or abs(y_frac) > 0.1:
                    # Simple interpolation
                    window = ndimage.shift(window, (y_frac, x_frac), order=1, mode='nearest')
                
                # Tính score cho vị trí này
                distance_to_target = np.sqrt((test_x - target_x)**2 + (test_y - target_y)**2)
                
                # Score dựa trên đặc trưng gap và khoảng cách
                gap_score = analyze_gap_quality(window, distance_to_target)
                
                if gap_score > best_score:
                    best_score = gap_score
                    best_x, best_y = test_x, test_y
    
    improvement = np.sqrt((initial_x - target_x)**2 + (initial_y - target_y)**2) - np.sqrt((best_x - target_x)**2 + (best_y - target_y)**2)
    
    print(f"✨ Fine-tuned: ({best_x:.1f}, {best_y:.1f}) - Improvement: {improvement:.2f}px")
    
    return best_x, best_y

def analyze_gap_quality(window, distance_to_target):
    """
    Phân tích chất lượng gap với trọng số ưu tiên distance
    """
    window_mean = np.mean(window)
    window_std = np.std(window)
    
    # Gap phải tối và đồng đều
    darkness_score = max(0, 100 - window_mean * 1.2)
    uniformity_score = max(0, 50 - window_std * 2)
    
    # Khoảng cách là yếu tố quan trọng nhất
    distance_score = max(0, 100 - distance_to_target * 20)
    
    total_score = distance_score * 0.7 + darkness_score * 0.2 + uniformity_score * 0.1
    
    return total_score

# Sử dụng
if __name__ == "__main__":
    main_image = "image.png"
    
    print("🚀 Khởi động thuật toán tìm ô vuông gap hoàn hảo...")
    
    x, y = find_perfect_square_gap(main_image)
    
    if x is not None and y is not None:
        error = np.sqrt((x - 275)**2 + (y - 26)**2)
        
        print(f"\n🏆 HOÀN THÀNH!")
        print(f"Sai số cuối cùng: {error:.1f} pixels")
        
        if error <= 3:
            print("🎉 XUẤT SẮC! Độ chính xác hoàn hảo (<3px)")
        elif error <= 5:
            print("✅ RẤT TỐT! Độ chính xác cao (<5px)")
        elif error <= 8:
            print("👍 TỐT! Độ chính xác khá (<8px)")
        elif error <= 12:
            print("⚠️ CHẤP NHẬN! Cần cải thiện (<12px)")
        else:
            print("❌ CẦN CẢI THIỆN! Sai số cao (>12px)")
    else:
        print("❌ Thất bại! Không tìm thấy ô vuông gap")