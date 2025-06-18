from PIL import Image, ImageDraw
import numpy as np
import cv2
from skimage import feature, filters, measure
from scipy import ndimage
import os
from datetime import datetime

def find_gap_by_template_matching(main_image_path):
    """
    Tìm gap bằng cách matching mảnh ghép với vùng trên
    """
    print("=== TÌM GAP BẰNG TEMPLATE MATCHING THÔNG MINH ===")
    
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    print(f"📏 Image size: {w}x{h}")
    
    # Bước 1: Tách mảnh ghép và vùng tìm kiếm
    piece_region, search_region, piece_info = extract_piece_and_search_area(gray)
    
    if piece_region is None:
        print("❌ Không tìm thấy mảnh ghép")
        return None, None
    
    print(f"🧩 Piece size: {piece_region.shape}")
    print(f"🔍 Search area size: {search_region.shape}")
    
    # Bước 2: Template matching để tìm vị trí khớp
    match_candidates = find_matching_positions(piece_region, search_region, piece_info)
    
    print(f"🎯 Found {len(match_candidates)} matching positions")
    
    # Bước 3: Kiểm tra gap tại mỗi vị trí match
    gap_candidates = []
    for match in match_candidates:
        gaps = find_gaps_near_match(gray, match, piece_info)
        gap_candidates.extend(gaps)
    
    print(f"🔍 Found {len(gap_candidates)} gap candidates")
    
    if not gap_candidates:
        print("❌ Không tìm thấy gap candidates")
        return None, None
    
    # Bước 4: Chọn gap tốt nhất
    best_gap = select_best_gap(gap_candidates, piece_info)
    
    # Bước 5: Visualize kết quả
    visualize_template_matching_result(main_image_path, piece_info, match_candidates, gap_candidates, best_gap)
    
    print(f"\n🎯 KẾT QUẢ TEMPLATE MATCHING:")
    print(f"Best gap position: ({best_gap['x']}, {best_gap['y']})")
    print(f"Gap score: {best_gap['score']:.1f}")
    print(f"Match confidence: {best_gap['match_confidence']:.3f}")
    
    return best_gap['x'], best_gap['y']

def extract_piece_and_search_area(gray):
    """
    Tách mảnh ghép và vùng tìm kiếm
    """
    h, w = gray.shape
    
    # Vùng mảnh ghép: 1/3 dưới của ảnh
    piece_start_y = h * 2 // 3
    piece_region = gray[piece_start_y:h, :]
    
    # Vùng tìm kiếm: 2/3 trên của ảnh  
    search_region = gray[0:piece_start_y, :]
    
    # Phân tích mảnh ghép để lấy template
    piece_info = analyze_piece_for_template(piece_region, piece_start_y)
    
    return piece_region, search_region, piece_info

def analyze_piece_for_template(piece_region, offset_y):
    """
    Phân tích mảnh ghép để tạo template
    """
    h, w = piece_region.shape
    
    # Tìm contour của mảnh ghép
    # Threshold để tách piece khỏi background - thử nhiều method
    mean_val = np.mean(piece_region)
    std_val = np.std(piece_region)
    
    print(f"🔍 Piece region stats: mean={mean_val:.1f}, std={std_val:.1f}")
    
    # Thử multiple thresholds
    binary_methods = [
        piece_region < (mean_val - std_val * 0.5),  # Method 1
        piece_region < (mean_val - 30),             # Method 2  
        piece_region > (mean_val + std_val * 0.5),  # Method 3 (bright piece)
    ]
    
    best_template = None
    best_area = 0
    
    for i, binary in enumerate(binary_methods):
        print(f"🔍 Trying threshold method {i+1}...")
        
        # Morphological operations
        from skimage import morphology
        binary_clean = morphology.opening(binary, morphology.disk(2))
        binary_clean = morphology.closing(binary_clean, morphology.disk(3))
        
        # Tìm connected components
        labeled = measure.label(binary_clean)
        regions = measure.regionprops(labeled)
        
        if regions:
            largest_region = max(regions, key=lambda x: x.area)
            print(f"   Found region with area: {largest_region.area}")
            
            if largest_region.area > best_area and largest_region.area > 500:  # Min area threshold
                best_area = largest_region.area
                bbox = largest_region.bbox
                
                # Extract template từ bbox
                template = piece_region[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                
                # Resize template nếu cần
                max_size = 50  # Smaller template for better matching
                if template.shape[0] > max_size or template.shape[1] > max_size:
                    scale_factor = min(max_size/template.shape[0], max_size/template.shape[1])
                    new_h = int(template.shape[0] * scale_factor)
                    new_w = int(template.shape[1] * scale_factor)
                    template = cv2.resize(template, (new_w, new_h))
                
                best_template = {
                    'template': template,
                    'bbox': bbox,
                    'offset_y': offset_y,
                    'area': largest_region.area,
                    'original_piece': piece_region,
                    'method': i+1
                }
    
    if best_template is None:
        print("⚠️ Không tìm thấy piece region phù hợp, dùng crop từ center")
        # Fallback: crop từ center của piece region
        center_h, center_w = h//2, w//2
        crop_size = 40
        template = piece_region[center_h-crop_size//2:center_h+crop_size//2,
                               center_w-crop_size//2:center_w+crop_size//2]
        
        best_template = {
            'template': template,
            'bbox': (center_h-crop_size//2, center_w-crop_size//2, center_h+crop_size//2, center_w+crop_size//2),
            'offset_y': offset_y,
            'area': crop_size * crop_size,
            'original_piece': piece_region,
            'method': 'fallback'
        }
    
    print(f"🧩 Selected template (method {best_template['method']}): {best_template['template'].shape}")
    print(f"📍 Piece bbox: {best_template['bbox']}")
    
    return best_template

def find_matching_positions(piece_region, search_region, piece_info):
    """
    Tìm vị trí matching giữa piece và search area
    """
    template = piece_info['template']
    search_h, search_w = search_region.shape
    template_h, template_w = template.shape
    
    if template_h >= search_h or template_w >= search_w:
        print("⚠️ Template quá lớn so với search area")
        return []
    
    # Template matching với OpenCV
    result = cv2.matchTemplate(search_region.astype(np.float32), 
                              template.astype(np.float32), 
                              cv2.TM_CCOEFF_NORMED)
    
    # Debug: in thông tin về template matching
    max_val = np.max(result)
    min_val = np.min(result)
    print(f"🔍 Template matching stats: min={min_val:.3f}, max={max_val:.3f}")
    
    # Tìm tất cả matches với threshold thấp hơn
    threshold = 0.3  # Threshold thấp hơn để tìm được matches
    locations = np.where(result >= threshold)
    
    matches = []
    for pt in zip(*locations[::-1]):  # (x, y)
        confidence = result[pt[1], pt[0]]
        
        # Tọa độ center của match
        center_x = pt[0] + template_w // 2
        center_y = pt[1] + template_h // 2
        
        matches.append({
            'x': center_x,
            'y': center_y,
            'confidence': confidence,
            'template_x': pt[0],
            'template_y': pt[1]
        })
    
    print(f"🔍 Found {len(matches)} raw matches with threshold {threshold}")
    
    # Loại bỏ duplicates (matches gần nhau)
    unique_matches = []
    for match in matches:
        is_duplicate = False
        for existing in unique_matches:
            distance = np.sqrt((match['x'] - existing['x'])**2 + (match['y'] - existing['y'])**2)
            if distance < 30:  # Nếu gần nhau < 30px
                # Giữ match có confidence cao hơn
                if match['confidence'] > existing['confidence']:
                    unique_matches.remove(existing)
                    unique_matches.append(match)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_matches.append(match)
    
    # Sort theo confidence
    unique_matches.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"🎯 Template matching results:")
    for i, match in enumerate(unique_matches[:5]):
        print(f"  {i+1}. ({match['x']}, {match['y']}) - Confidence: {match['confidence']:.3f}")
    
    return unique_matches

def find_gaps_near_match(gray, match, piece_info):
    """
    Tìm gaps gần vị trí match
    """
    h, w = gray.shape
    match_x, match_y = match['x'], match['y']
    
    # Tìm gaps trong vùng quanh match
    search_radius = 50
    gap_candidates = []
    
    # Kích thước gap dự kiến (dựa trên piece)
    template = piece_info['template']
    expected_gap_size = min(template.shape[0], template.shape[1])
    
    window_sizes = [
        max(16, expected_gap_size - 8),
        expected_gap_size,
        min(50, expected_gap_size + 8)
    ]
    
    for window_size in window_sizes:
        half_size = window_size // 2
        
        for dy in range(-search_radius, search_radius + 1, 4):
            for dx in range(-search_radius, search_radius + 1, 4):
                
                gap_x = match_x + dx
                gap_y = match_y + dy
                
                # Check bounds
                if (gap_x - half_size < 0 or gap_x + half_size >= w or
                    gap_y - half_size < 0 or gap_y + half_size >= h):
                    continue
                
                # Extract window
                window = gray[gap_y - half_size:gap_y + half_size,
                             gap_x - half_size:gap_x + half_size]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    continue
                
                # Analyze gap
                gap_score = analyze_gap_near_match(window, gap_x, gap_y, match, piece_info)
                
                if gap_score > 70:  # Threshold cao cho gap chất lượng
                    distance_to_match = np.sqrt((gap_x - match_x)**2 + (gap_y - match_y)**2)
                    
                    gap_candidates.append({
                        'x': gap_x,
                        'y': gap_y,
                        'score': gap_score,
                        'window_size': window_size,
                        'distance_to_match': distance_to_match,
                        'match_confidence': match['confidence'],
                        'match_x': match_x,
                        'match_y': match_y
                    })
    
    return gap_candidates

def analyze_gap_near_match(window, gap_x, gap_y, match, piece_info):
    """
    Phân tích gap gần vị trí match
    """
    window_size = window.shape[0]
    window_mean = np.mean(window)
    window_std = np.std(window)
    
    # Background comparison (lấy từ các vùng xung quanh)
    bg_sample = piece_info['original_piece'][:20, :20]  # Sample từ piece region
    bg_mean = np.mean(bg_sample)
    
    # 1. Gap phải tối hơn background
    darkness_ratio = (bg_mean - window_mean) / bg_mean if bg_mean > 0 else 0
    darkness_score = max(0, min(100, darkness_ratio * 300))
    
    # 2. Gap phải đồng đều
    uniformity_score = max(0, 100 - window_std * 2)
    
    # 3. Hình dạng vuông
    row_means = [np.mean(window[i, :]) for i in range(window_size)]
    col_means = [np.mean(window[:, j]) for j in range(window_size)]
    shape_score = max(0, 100 - np.std(row_means) * 5 - np.std(col_means) * 5)
    
    # 4. Khoảng cách đến match (gần match tốt hơn)
    distance_to_match = np.sqrt((gap_x - match['x'])**2 + (gap_y - match['y'])**2)
    distance_score = max(0, 100 - distance_to_match * 2)
    
    # 5. Confidence của match
    match_score = match['confidence'] * 100
    
    # Tổng hợp
    total_score = (
        darkness_score * 0.25 +
        uniformity_score * 0.25 +
        shape_score * 0.20 +
        distance_score * 0.15 +
        match_score * 0.15
    )
    
    return total_score

def select_best_gap(gap_candidates, piece_info):
    """
    Chọn gap tốt nhất
    """
    # Sort theo score tổng hợp
    gap_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    print(f"🏆 Top 5 gap candidates:")
    for i, gap in enumerate(gap_candidates[:5]):
        print(f"  {i+1}. ({gap['x']}, {gap['y']}) - Score: {gap['score']:.1f} - Match conf: {gap['match_confidence']:.3f}")
    
    return gap_candidates[0]

def visualize_template_matching_result(main_image_path, piece_info, matches, gaps, best_gap):
    """
    Visualize kết quả template matching
    """
    try:
        img = Image.open(main_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Vẽ piece region
        piece_bbox = piece_info['bbox']
        offset_y = piece_info['offset_y']
        draw.rectangle([
            piece_bbox[1], offset_y + piece_bbox[0],
            piece_bbox[3], offset_y + piece_bbox[2]
        ], outline=(255, 255, 0), width=2)  # Vàng cho piece
        
        # Vẽ matches
        for i, match in enumerate(matches[:3]):  # Top 3 matches
            color = (0, 255, 0) if i == 0 else (0, 255, 255)  # Xanh lá cho best, cyan cho others
            x, y = match['x'], match['y']
            draw.ellipse([x-8, y-8, x+8, y+8], outline=color, width=2)
            draw.text((x+10, y-10), f"M{i+1}", fill=color)
        
        # Vẽ gap candidates
        for i, gap in enumerate(gaps[:10]):
            if gap == best_gap:
                color = (255, 0, 0)  # Đỏ cho best gap
                radius = 12
            else:
                color = (255, 165, 0)  # Cam cho other gaps
                radius = 6
            
            x, y = gap['x'], gap['y']
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        outline=color, width=3)
        
        # Info text
        info_text = [
            f"Best Gap: ({best_gap['x']}, {best_gap['y']})",
            f"Score: {best_gap['score']:.1f}",
            f"Match Conf: {best_gap['match_confidence']:.3f}",
            f"Template Size: {piece_info['template'].shape}"
        ]
        
        for i, text in enumerate(info_text):
            draw.rectangle([10, 10+i*20, 300, 25+i*20], fill=(0, 0, 0), outline=(255, 255, 255))
            draw.text((12, 12+i*20), text, fill=(255, 255, 255))
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/template_matching_{timestamp}.png"
        if not os.path.exists("results"):
            os.makedirs("results")
        img.save(output_path)
        
        print(f"✅ Saved template matching result: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")

# Sử dụng
if __name__ == "__main__":
    main_image = "image.png"
    
    print("🚀 Khởi động template matching thông minh...")
    
    x, y = find_gap_by_template_matching(main_image)
    
    if x is not None and y is not None:
        print(f"\n🏆 THÀNH CÔNG!")
        print(f"Gap position found: ({x}, {y})")
        print(f"📸 Xem file visualization để kiểm tra kết quả!")
    else:
        print("❌ Thất bại! Không tìm thấy gap")
