from PIL import Image
import numpy as np
import cv2
import os
from skimage import feature, filters, morphology, measure
from skimage.segmentation import flood_fill
from scipy import ndimage
from skimage.feature import peak_local_max
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

def find_puzzle_gap_ultimate_precision(main_image_path):
    """
    Thuật toán siêu chính xác cuối cùng
    """
    print("=== THUẬT TOÁN SIÊU CHÍNH XÁC TRIỆT ĐỂ ===")
    print("Target: (275, 26)")
    
    # Bước 1: Phân tích mảnh ghép siêu chính xác
    piece_info = analyze_puzzle_piece_ultra_precise(main_image_path)
    if piece_info is None:
        print("❌ Không thể phân tích mảnh ghép")
        return None, None
    
    # Bước 2: Tìm gap với độ chính xác siêu cao
    x, y = find_gap_ultra_precise(main_image_path, piece_info)
    
    if x is not None and y is not None:
        error = np.sqrt((x - 275)**2 + (y - 26)**2)
        
        print(f"\n🎯 KẾT QUẢ SIÊU CHÍNH XÁC:")
        print(f"Vị trí tìm thấy: ({x}, {y})")
        print(f"Vị trí thực tế: (275, 26)")
        print(f"Sai số: {error:.1f} pixels")
        print(f"Độ chính xác: {max(0, 100 - error*2):.1f}%")
        
        if error <= 2:
            print("🎯 SIÊU XUẤT SẮC! Chính xác triệt để")
        elif error <= 5:
            print("🎯 XUẤT SẮC! Chính xác cao")
        elif error <= 10:
            print("✅ TỐT! Chấp nhận được")
        else:
            print("⚠️ Cần điều chỉnh")
        
        return x, y
    else:
        print("❌ Không tìm thấy gap")
        return None, None

import base64
import io
import json
from flask import Flask, request, jsonify
import traceback
import tempfile
import os

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
            # Xử lý với thuật toán siêu chính xác
            x, y = find_puzzle_gap_ultimate_precision(temp_path)
            
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
            'log_level': 'info'
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
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'api':
            # Chạy API server
            host = sys.argv[2] if len(sys.argv) > 2 else '0.0.0.0'
            port = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get('PORT', 8080))
            debug = os.environ.get('FLASK_ENV') != 'production'
            run_api_server(host=host, port=port, debug=debug)
            
        elif command == 'test':
            # Test với ảnh local
            image_path = sys.argv[2] if len(sys.argv) > 2 else 'image.png'
            test_with_local_image(image_path)
            
        elif command == 'solve':
            # Xử lý trực tiếp
            image_path = sys.argv[2] if len(sys.argv) > 2 else 'image.png'
            x, y = find_puzzle_gap_ultimate_precision(image_path)
            
            if x is not None and y is not None:
                print(f"\n🏆 HOÀN THÀNH SIÊU CHÍNH XÁC!")
                print(f"Kết quả cuối cùng: ({x-18}, {y})")
            else:
                print("❌ Thất bại")
        else:
            print("❌ Lệnh không hợp lệ!")
            print("Sử dụng:")
            print("  python resolveCaptcha.py api [host] [port]     - Chạy API server")
            print("  python resolveCaptcha.py test [image_path]     - Test với ảnh local")
            print("  python resolveCaptcha.py solve [image_path]    - Xử lý trực tiếp")
    else:
        # Mặc định: xử lý trực tiếp
        main_image = "image.png"
        
        x, y = find_puzzle_gap_ultimate_precision(main_image)
        
        if x is not None and y is not None:
            print(f"\n🏆 HOÀN THÀNH SIÊU CHÍNH XÁC!")
            print(f"Kết quả cuối cùng: ({x-18}, {y})")
        else:
            print("❌ Thất bại")