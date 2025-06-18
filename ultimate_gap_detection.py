#!/usr/bin/env python3

from PIL import Image, ImageDraw
import numpy as np
import cv2
from skimage import feature, filters, measure, morphology
from scipy import ndimage
import os
from datetime import datetime

def find_gap_ultimate(main_image_path):
    """
    Phiên bản ultimate của thuật toán tìm gap
    Kết hợp template matching + shape analysis + context awareness
    """
    print("=== ULTIMATE GAP DETECTION ===")
    
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    print(f"📏 Image size: {w}x{h}")
    
    # Bước 1: Phân tích cấu trúc puzzle
    puzzle_info = analyze_puzzle_structure(gray)
    
    # Bước 2: Extract mảnh ghép thông minh
    piece_info = extract_piece_smart(gray, puzzle_info)
    
    if piece_info is None:
        print("❌ Không tìm thấy mảnh ghép")
        return None, None
    
    # Bước 3: Template matching với context
    matches = template_matching_with_context(gray, piece_info, puzzle_info)
    
    # Bước 4: Verify gaps bằng shape analysis
    verified_gaps = verify_gaps_by_shape(gray, matches, piece_info)
    
    # Bước 5: Select best gap với comprehensive scoring
    best_gap = ultimate_gap_selection(verified_gaps, piece_info, puzzle_info)
    
    if best_gap is None:
        print("❌ Không tìm thấy gap hợp lệ")
        return None, None
    
    # Visualize
    visualize_ultimate_result(main_image_path, piece_info, matches, verified_gaps, best_gap)
    
    print(f"\n🏆 ULTIMATE RESULT:")
    print(f"Gap position: ({best_gap['x']}, {best_gap['y']})")
    print(f"Confidence: {best_gap['confidence']:.3f}")
    print(f"Shape match: {best_gap['shape_score']:.1f}")
    
    return best_gap['x'], best_gap['y']

def analyze_puzzle_structure(gray):
    """
    Phân tích cấu trúc tổng thể của puzzle
    """
    h, w = gray.shape
    
    # Phân tích gradient để tìm edges
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Phân tích brightness distribution
    top_half = gray[:h//2, :]
    bottom_half = gray[h//2:, :]
    
    puzzle_info = {
        'width': w,
        'height': h,
        'avg_brightness': np.mean(gray),
        'top_brightness': np.mean(top_half),
        'bottom_brightness': np.mean(bottom_half),
        'edge_density': np.mean(edges),
        'piece_area_y': h * 2 // 3,  # Vùng piece bắt đầu
    }
    
    print(f"🧩 Puzzle structure analysis:")
    print(f"  Average brightness: {puzzle_info['avg_brightness']:.1f}")
    print(f"  Top/Bottom ratio: {puzzle_info['top_brightness']/puzzle_info['bottom_brightness']:.2f}")
    print(f"  Edge density: {puzzle_info['edge_density']:.1f}")
    
    return puzzle_info

def extract_piece_smart(gray, puzzle_info):
    """
    Extract mảnh ghép thông minh hơn
    """
    h, w = gray.shape
    piece_start_y = puzzle_info['piece_area_y']
    
    piece_region = gray[piece_start_y:h, :]
    print(f"🧩 Piece region: {piece_region.shape}")
    
    # Multiple methods để extract piece shape
    methods = [
        extract_by_threshold,
        extract_by_edge_detection,
        extract_by_morphology,
    ]
    
    best_piece = None
    best_score = 0
    
    for method in methods:
        try:
            piece_candidate = method(piece_region, puzzle_info)
            if piece_candidate:
                score = evaluate_piece_quality(piece_candidate)
                print(f"  Method {method.__name__}: score={score:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_piece = piece_candidate
                    best_piece['method'] = method.__name__
                    best_piece['offset_y'] = piece_start_y
        except Exception as e:
            print(f"  Method {method.__name__} failed: {e}")
    
    if best_piece:
        print(f"🏆 Best piece extraction: {best_piece['method']} (score={best_score:.1f})")
    
    return best_piece

def extract_by_threshold(piece_region, puzzle_info):
    """
    Extract piece bằng threshold adaptive
    """
    mean_val = np.mean(piece_region)
    std_val = np.std(piece_region)
    
    # Dynamic threshold dựa trên brightness pattern
    if puzzle_info['bottom_brightness'] > puzzle_info['top_brightness']:
        # Bright piece on dark background
        threshold = mean_val + std_val * 0.3
        binary = piece_region > threshold
    else:
        # Dark piece on bright background  
        threshold = mean_val - std_val * 0.3
        binary = piece_region < threshold
    
    # Clean up binary image
    binary = morphology.opening(binary, morphology.disk(3))
    binary = morphology.closing(binary, morphology.disk(5))
    
    # Find largest component
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return None
    
    largest_region = max(regions, key=lambda x: x.area)
    bbox = largest_region.bbox
    
    # Extract template
    template = piece_region[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    return {
        'template': template,
        'bbox': bbox,
        'binary': binary,
        'area': largest_region.area,
        'centroid': largest_region.centroid
    }

def extract_by_edge_detection(piece_region, puzzle_info):
    """
    Extract piece bằng edge detection
    """
    # Canny edge detection
    edges = cv2.Canny((piece_region * 255).astype(np.uint8), 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)
    bbox = (y, x, y+h, x+w)
    
    # Template
    template = piece_region[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    return {
        'template': template,
        'bbox': bbox,
        'contour': largest_contour,
        'area': cv2.contourArea(largest_contour)
    }

def extract_by_morphology(piece_region, puzzle_info):
    """
    Extract piece bằng morphological operations
    """
    # Adaptive threshold
    mean_val = np.mean(piece_region)
    binary = piece_region < mean_val
    
    # Morphological operations
    kernel = morphology.disk(4)
    binary = morphology.opening(binary, kernel)
    binary = morphology.closing(binary, kernel)
    binary = morphology.remove_small_objects(binary, min_size=500)
    
    # Find regions
    labeled = measure.label(binary)
    regions = measure.regionprops(labeled)
    
    if not regions:
        return None
    
    # Best region (largest with good shape)
    best_region = None
    best_compactness = 0
    
    for region in regions:
        if region.area > 1000:  # Minimum size
            # Compactness = area / (perimeter^2)
            compactness = 4 * np.pi * region.area / (region.perimeter ** 2)
            if compactness > best_compactness:
                best_compactness = compactness
                best_region = region
    
    if not best_region:
        return max(regions, key=lambda x: x.area)
    
    bbox = best_region.bbox
    template = piece_region[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    
    return {
        'template': template,
        'bbox': bbox,
        'binary': binary,
        'area': best_region.area,
        'compactness': best_compactness
    }

def evaluate_piece_quality(piece_info):
    """
    Đánh giá chất lượng piece extraction
    """
    template = piece_info['template']
    
    if template.size == 0:
        return 0
    
    # Size score
    size_score = min(100, template.size / 100)
    
    # Shape score (aspect ratio)
    h, w = template.shape
    aspect_ratio = min(w/h, h/w)
    shape_score = aspect_ratio * 100
    
    # Contrast score
    contrast_score = np.std(template) * 2
    
    # Area score
    area_score = min(100, piece_info.get('area', 0) / 100)
    
    total_score = (size_score * 0.3 + shape_score * 0.3 + 
                   contrast_score * 0.2 + area_score * 0.2)
    
    return total_score

def template_matching_with_context(gray, piece_info, puzzle_info):
    """
    Template matching với context awareness
    """
    h, w = gray.shape
    template = piece_info['template']
    
    # Search area - chỉ tìm ở phần trên
    search_area = gray[:puzzle_info['piece_area_y'], :]
    
    # Multiple template matching methods
    methods = [
        cv2.TM_CCOEFF_NORMED,
        cv2.TM_CCORR_NORMED,
        cv2.TM_SQDIFF_NORMED,
    ]
    
    all_matches = []
    
    for method in methods:
        result = cv2.matchTemplate(search_area.astype(np.float32), 
                                  template.astype(np.float32), method)
        
        # Adaptive threshold
        if method == cv2.TM_SQDIFF_NORMED:
            threshold = 0.7  # Lower is better for SQDIFF
            locations = np.where(result <= threshold)
            confidences = 1 - result[locations]  # Invert for SQDIFF
        else:
            threshold = 0.3
            locations = np.where(result >= threshold)
            confidences = result[locations]
        
        for i, (y, x) in enumerate(zip(*locations)):
            match = {
                'x': x + template.shape[1] // 2,
                'y': y + template.shape[0] // 2,
                'confidence': confidences[i],
                'method': method,
                'template_x': x,
                'template_y': y
            }
            all_matches.append(match)
    
    # Remove duplicates and sort
    unique_matches = remove_duplicate_matches(all_matches)
    
    print(f"🎯 Found {len(unique_matches)} unique matches from all methods")
    
    return unique_matches

def remove_duplicate_matches(matches):
    """
    Remove duplicate matches
    """
    unique_matches = []
    
    for match in matches:
        is_duplicate = False
        for existing in unique_matches:
            distance = np.sqrt((match['x'] - existing['x'])**2 + 
                             (match['y'] - existing['y'])**2)
            if distance < 25:
                # Keep higher confidence
                if match['confidence'] > existing['confidence']:
                    unique_matches.remove(existing)
                    unique_matches.append(match)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_matches.append(match)
    
    return sorted(unique_matches, key=lambda x: x['confidence'], reverse=True)

def verify_gaps_by_shape(gray, matches, piece_info):
    """
    Verify gaps bằng shape analysis
    """
    verified_gaps = []
    template_shape = piece_info['template'].shape
    
    for match in matches:
        # Check for gap near match
        gaps_near_match = find_gaps_with_shape_analysis(gray, match, template_shape)
        
        for gap in gaps_near_match:
            # Shape matching score
            shape_score = calculate_shape_match_score(gray, gap, piece_info)
            
            gap['shape_score'] = shape_score
            gap['match_info'] = match
            
            if shape_score > 60:  # Threshold for good shape match
                verified_gaps.append(gap)
    
    print(f"🔍 Verified {len(verified_gaps)} gaps by shape analysis")
    
    return verified_gaps

def find_gaps_with_shape_analysis(gray, match, expected_shape):
    """
    Tìm gaps với shape analysis
    """
    h, w = gray.shape
    match_x, match_y = match['x'], match['y']
    
    gaps = []
    search_radius = 40
    
    # Expected gap size
    expected_h, expected_w = expected_shape
    
    for window_size in [max(16, min(expected_h, expected_w) - 4),
                       min(expected_h, expected_w),
                       min(50, max(expected_h, expected_w) + 4)]:
        
        half_size = window_size // 2
        
        for dy in range(-search_radius, search_radius + 1, 6):
            for dx in range(-search_radius, search_radius + 1, 6):
                
                gap_x = match_x + dx
                gap_y = match_y + dy
                
                if (gap_x - half_size < 0 or gap_x + half_size >= w or
                    gap_y - half_size < 0 or gap_y + half_size >= h):
                    continue
                
                window = gray[gap_y - half_size:gap_y + half_size,
                             gap_x - half_size:gap_x + half_size]
                
                # Basic gap analysis
                gap_quality = analyze_gap_quality_advanced(window, gap_x, gap_y, match)
                
                if gap_quality > 75:
                    gaps.append({
                        'x': gap_x,
                        'y': gap_y,
                        'quality': gap_quality,
                        'window_size': window_size,
                        'distance_to_match': np.sqrt(dx**2 + dy**2)
                    })
    
    return gaps

def analyze_gap_quality_advanced(window, gap_x, gap_y, match):
    """
    Advanced gap quality analysis
    """
    # Basic stats
    mean_val = np.mean(window)
    std_val = np.std(window)
    
    # 1. Darkness (gaps should be dark)
    darkness_score = max(0, (255 - mean_val) / 255 * 100)
    
    # 2. Uniformity (gaps should be uniform)
    uniformity_score = max(0, 100 - std_val * 3)
    
    # 3. Shape regularity (gaps should be square-like)
    h, w = window.shape
    aspect_ratio = min(w/h, h/w)
    shape_score = aspect_ratio * 100
    
    # 4. Edge strength (gaps should have clear edges)
    edges = cv2.Canny((window * 255).astype(np.uint8), 50, 150)
    edge_score = min(100, np.sum(edges > 0) / window.size * 200)
    
    # 5. Distance to match (closer is better)
    distance_score = max(0, 100 - np.sqrt((gap_x - match['x'])**2 + (gap_y - match['y'])**2) * 2)
    
    # Weighted combination
    total_score = (
        darkness_score * 0.3 +
        uniformity_score * 0.25 +
        shape_score * 0.2 +
        edge_score * 0.1 +
        distance_score * 0.15
    )
    
    return total_score

def calculate_shape_match_score(gray, gap, piece_info):
    """
    Calculate shape matching score between gap and piece
    """
    template = piece_info['template']
    gap_x, gap_y = gap['x'], gap['y']
    window_size = gap['window_size']
    
    half_size = window_size // 2
    
    # Extract gap window
    gap_window = gray[gap_y - half_size:gap_y + half_size,
                     gap_x - half_size:gap_x + half_size]
    
    # Resize template to match gap window
    template_resized = cv2.resize(template, (window_size, window_size))
    
    # Template matching between gap and piece
    result = cv2.matchTemplate(gap_window.astype(np.float32),
                              template_resized.astype(np.float32),
                              cv2.TM_CCOEFF_NORMED)
    
    max_correlation = np.max(result)
    
    # Shape correlation score
    shape_score = max(0, max_correlation * 100)
    
    return shape_score

def ultimate_gap_selection(verified_gaps, piece_info, puzzle_info):
    """
    Ultimate gap selection với comprehensive scoring
    """
    if not verified_gaps:
        return None
    
    # Score mỗi gap
    for gap in verified_gaps:
        # Combine all scores
        quality_score = gap['quality']
        shape_score = gap['shape_score']
        match_confidence = gap['match_info']['confidence'] * 100
        
        # Distance penalty (closer to expected position is better)
        expected_x = puzzle_info['width'] // 2
        expected_y = puzzle_info['height'] // 3
        distance_penalty = np.sqrt((gap['x'] - expected_x)**2 + (gap['y'] - expected_y)**2) * 0.1
        
        # Final comprehensive score
        gap['final_score'] = (
            quality_score * 0.4 +
            shape_score * 0.3 +
            match_confidence * 0.2 +
            max(0, 100 - distance_penalty) * 0.1
        )
        
        gap['confidence'] = min(1.0, gap['final_score'] / 100)
    
    # Sort by final score
    verified_gaps.sort(key=lambda x: x['final_score'], reverse=True)
    
    print(f"🏆 Top 3 ultimate gaps:")
    for i, gap in enumerate(verified_gaps[:3]):
        print(f"  {i+1}. ({gap['x']}, {gap['y']}) - Final score: {gap['final_score']:.1f}")
    
    return verified_gaps[0]

def visualize_ultimate_result(main_image_path, piece_info, matches, verified_gaps, best_gap):
    """
    Visualize ultimate result
    """
    try:
        img = Image.open(main_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Draw piece
        piece_bbox = piece_info['bbox']
        offset_y = piece_info['offset_y']
        draw.rectangle([
            piece_bbox[1], offset_y + piece_bbox[0],
            piece_bbox[3], offset_y + piece_bbox[2]
        ], outline=(255, 255, 0), width=2)
        
        # Draw matches
        for i, match in enumerate(matches[:5]):
            color = (0, 255, 0) if i == 0 else (0, 255, 255)
            x, y = match['x'], match['y']
            draw.ellipse([x-6, y-6, x+6, y+6], outline=color, width=2)
        
        # Draw verified gaps
        for gap in verified_gaps:
            if gap == best_gap:
                color = (255, 0, 0)
                radius = 15
            else:
                color = (255, 165, 0)
                radius = 8
            
            x, y = gap['x'], gap['y']
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        outline=color, width=3)
        
        # Info
        info_text = [
            f"ULTIMATE GAP DETECTION",
            f"Position: ({best_gap['x']}, {best_gap['y']})",
            f"Confidence: {best_gap['confidence']:.3f}",
            f"Final Score: {best_gap['final_score']:.1f}",
            f"Shape Score: {best_gap['shape_score']:.1f}"
        ]
        
        for i, text in enumerate(info_text):
            draw.rectangle([10, 10+i*25, 350, 30+i*25], fill=(0, 0, 0), outline=(255, 255, 255))
            draw.text((12, 12+i*25), text, fill=(255, 255, 255))
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/ultimate_gap_{timestamp}.png"
        if not os.path.exists("results"):
            os.makedirs("results")
        img.save(output_path)
        
        print(f"✅ Saved ultimate result: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")

if __name__ == "__main__":
    main_image = "image.png"
    
    print("🚀 Starting ULTIMATE gap detection...")
    
    x, y = find_gap_ultimate(main_image)
    
    if x is not None and y is not None:
        print(f"\n🏆 ULTIMATE SUCCESS!")
        print(f"🎯 Final gap position: ({x}, {y})")
        print(f"📸 Check visualization for verification!")
    else:
        print("❌ Ultimate detection failed!")
