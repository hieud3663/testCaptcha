from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import feature, filters, morphology, measure
import cv2
from scipy import ndimage
import os
from datetime import datetime

def analyze_debug_and_fix_scoring(main_image_path):
    """
    Phân tích lại debug candidates và cải thiện scoring
    """
    print("=== PHÂN TÍCH DEBUG VÀ SỬA SCORING ===")
    
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    print(f"📏 Image size: {w}x{h}")
    
    # Tìm TẤT CẢ candidates với threshold thấp
    all_candidates = find_all_gap_candidates(gray)
    
    print(f"🔍 Found {len(all_candidates)} total candidates")
    
    # Phân tích chi tiết từng candidate
    detailed_candidates = []
    for i, candidate in enumerate(all_candidates):
        x, y = candidate['x'], candidate['y']
        
        # Phân tích chi tiết candidate này
        detailed_analysis = deep_analyze_candidate(gray, x, y, candidate['window_size'])
        detailed_analysis.update(candidate)
        detailed_candidates.append(detailed_analysis)
    
    # Sort bằng nhiều criteria khác nhau
    print("\n🎯 PHÂN TÍCH CÁC PHƯƠNG PHÁP SCORING:")
    
    # Method 1: Original scoring  
    candidates_original = sorted(detailed_candidates, key=lambda x: x['original_score'], reverse=True)
    print(f"Method 1 (Original): Top candidate = ({candidates_original[0]['x']}, {candidates_original[0]['y']})")
    
    # Method 2: Darkness priority
    candidates_darkness = sorted(detailed_candidates, key=lambda x: x['darkness_score'], reverse=True)
    print(f"Method 2 (Darkness): Top candidate = ({candidates_darkness[0]['x']}, {candidates_darkness[0]['y']})")
    
    # Method 3: Uniformity priority  
    candidates_uniformity = sorted(detailed_candidates, key=lambda x: x['uniformity_score'], reverse=True)
    print(f"Method 3 (Uniformity): Top candidate = ({candidates_uniformity[0]['x']}, {candidates_uniformity[0]['y']})")
    
    # Method 4: Contrast priority
    candidates_contrast = sorted(detailed_candidates, key=lambda x: x['contrast_score'], reverse=True)
    print(f"Method 4 (Contrast): Top candidate = ({candidates_contrast[0]['x']}, {candidates_contrast[0]['y']})")
    
    # Method 5: Shape priority
    candidates_shape = sorted(detailed_candidates, key=lambda x: x['shape_score'], reverse=True)
    print(f"Method 5 (Shape): Top candidate = ({candidates_shape[0]['x']}, {candidates_shape[0]['y']})")
    
    # Method 6: Combined new scoring
    for candidate in detailed_candidates:
        candidate['new_score'] = calculate_new_score(candidate)
    
    candidates_new = sorted(detailed_candidates, key=lambda x: x['new_score'], reverse=True)
    print(f"Method 6 (New Combined): Top candidate = ({candidates_new[0]['x']}, {candidates_new[0]['y']})")
    
    # Visualize top candidates từ mỗi method
    visualize_multi_method_results(main_image_path, {
        'Original': candidates_original[0],
        'Darkness': candidates_darkness[0], 
        'Uniformity': candidates_uniformity[0],
        'Contrast': candidates_contrast[0],
        'Shape': candidates_shape[0],
        'New Combined': candidates_new[0]
    })
    
    # In detailed analysis của top 10 candidates
    print(f"\n📊 DETAILED ANALYSIS - TOP 10 CANDIDATES:")
    top_10 = sorted(detailed_candidates, key=lambda x: x['new_score'], reverse=True)[:10]
    
    for i, candidate in enumerate(top_10):
        print(f"\n{i+1}. Position: ({candidate['x']}, {candidate['y']})")
        print(f"   Window Size: {candidate['window_size']}")
        print(f"   Original Score: {candidate['original_score']:.1f}")
        print(f"   New Score: {candidate['new_score']:.1f}")
        print(f"   Darkness: {candidate['darkness_score']:.1f}")
        print(f"   Uniformity: {candidate['uniformity_score']:.1f}")
        print(f"   Contrast: {candidate['contrast_score']:.1f}")
        print(f"   Shape: {candidate['shape_score']:.1f}")
        print(f"   Edge: {candidate['edge_score']:.1f}")
    
    return candidates_new[0]

def find_all_gap_candidates(gray):
    """
    Tìm TẤT CẢ candidates với threshold thấp
    """
    h, w = gray.shape
    candidates = []
    window_sizes = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    
    # Background stats
    bg_mean = np.mean(gray[:50, :50])
    
    # Scan toàn bộ ảnh với step nhỏ
    for window_size in window_sizes:
        half_size = window_size // 2
        
        for y in range(half_size, h - half_size, 6):  # Step = 6px
            for x in range(half_size, w - half_size, 6):
                
                window = gray[y - half_size:y + half_size,
                             x - half_size:x + half_size]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    continue
                
                # Basic scoring để filter
                window_mean = np.mean(window)
                darkness_ratio = (bg_mean - window_mean) / bg_mean if bg_mean > 0 else 0
                
                # Threshold rất thấp để không bỏ sót
                if darkness_ratio > 0.1:  # Chỉ cần tối hơn bg 10%
                    candidates.append({
                        'x': x,
                        'y': y,
                        'window_size': window_size,
                        'basic_darkness': darkness_ratio * 100
                    })
    
    return candidates

def deep_analyze_candidate(gray, x, y, window_size):
    """
    Phân tích chi tiết một candidate
    """
    h, w = gray.shape
    half_size = window_size // 2
    
    # Extract window
    window = gray[y - half_size:y + half_size, x - half_size:x + half_size]
    
    # Background comparison
    bg_regions = []
    if x >= 50 and y >= 50:
        bg_regions.append(gray[y-50:y-30, x-50:x-30])  # Top-left sample
    if x < w-50 and y >= 50:
        bg_regions.append(gray[y-50:y-30, x+30:x+50])  # Top-right sample
    
    bg_mean = np.mean([np.mean(region) for region in bg_regions]) if bg_regions else np.mean(gray[:50, :50])
    
    # Detailed analysis
    window_mean = np.mean(window)
    window_std = np.std(window)
    
    # 1. Darkness score
    darkness_ratio = (bg_mean - window_mean) / bg_mean if bg_mean > 0 else 0
    darkness_score = max(0, min(100, darkness_ratio * 400))
    
    # 2. Uniformity score
    uniformity_score = max(0, 100 - window_std * 2)
    
    # 3. Shape score (square-ness)
    row_means = [np.mean(window[i, :]) for i in range(window_size)]
    col_means = [np.mean(window[:, j]) for j in range(window_size)]
    row_consistency = max(0, 100 - np.std(row_means) * 8)
    col_consistency = max(0, 100 - np.std(col_means) * 8)
    shape_score = (row_consistency + col_consistency) / 2
    
    # 4. Contrast score
    border_pixels = np.concatenate([
        window[0, :], window[-1, :], window[:, 0], window[:, -1]
    ])
    border_mean = np.mean(border_pixels)
    center_size = max(4, window_size // 2)
    start = (window_size - center_size) // 2
    center_region = window[start:start+center_size, start:start+center_size]
    center_mean = np.mean(center_region)
    contrast_score = max(0, min(100, (border_mean - center_mean) / max(border_mean, 1) * 150))
    
    # 5. Edge score
    grad_x = np.gradient(window, axis=1)
    grad_y = np.gradient(window, axis=0)
    edges = np.sqrt(grad_x**2 + grad_y**2)
    edge_score = min(100, np.mean(edges) * 10)
    
    # 6. Original score (như thuật toán cũ)
    size_score = 100 if 20 <= window_size <= 30 else 80 if 16 <= window_size <= 34 else 60
    
    original_score = (
        darkness_score * 0.25 +
        uniformity_score * 0.20 +
        shape_score * 0.20 +
        contrast_score * 0.20 +
        size_score * 0.15
    )
    
    return {
        'darkness_score': darkness_score,
        'uniformity_score': uniformity_score,
        'shape_score': shape_score,
        'contrast_score': contrast_score,
        'edge_score': edge_score,
        'original_score': original_score,
        'window_mean': window_mean,
        'bg_mean': bg_mean
    }

def calculate_new_score(candidate):
    """
    Scoring mới với trọng số khác
    """
    # New weighting - ưu tiên uniformity và darkness
    new_score = (
        candidate['uniformity_score'] * 0.35 +  # 35% - Uniformity quan trọng nhất
        candidate['darkness_score'] * 0.30 +    # 30% - Darkness
        candidate['contrast_score'] * 0.20 +    # 20% - Contrast
        candidate['shape_score'] * 0.10 +       # 10% - Shape
        candidate['edge_score'] * 0.05          # 5% - Edge
    )
    
    # Bonus cho gaps có đặc trưng rất rõ ràng
    if (candidate['uniformity_score'] > 80 and 
        candidate['darkness_score'] > 60 and 
        candidate['contrast_score'] > 50):
        new_score *= 1.4
    
    return new_score

def visualize_multi_method_results(main_image_path, method_results):
    """
    Visualize kết quả từ nhiều methods
    """
    try:
        img = Image.open(main_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        colors = {
            'Original': (255, 0, 0),      # Đỏ
            'Darkness': (0, 255, 0),      # Xanh lá
            'Uniformity': (0, 0, 255),    # Xanh dương  
            'Contrast': (255, 255, 0),    # Vàng
            'Shape': (255, 0, 255),       # Tím
            'New Combined': (0, 255, 255) # Cyan
        }
        
        # Vẽ results
        for i, (method, candidate) in enumerate(method_results.items()):
            color = colors[method]
            x, y = candidate['x'], candidate['y']
            
            # Vẽ circle với size khác nhau
            radius = 8 + i * 2
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        outline=color, width=3)
            
            # Label
            draw.text((x+radius+2, y-radius+i*3), method[:3], fill=color)
        
        # Legend
        legend_y = 10
        for method, color in colors.items():
            draw.rectangle([10, legend_y, 30, legend_y+15], fill=color)
            draw.text((35, legend_y+2), method, fill=(255, 255, 255))
            legend_y += 20
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/multi_method_debug_{timestamp}.png"
        img.save(output_path)
        
        print(f"✅ Saved multi-method visualization: {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving visualization: {e}")

# Sử dụng
if __name__ == "__main__":
    main_image = "image.png"
    
    print("🔍 Phân tích debug và cải thiện scoring...")
    
    best_candidate = analyze_debug_and_fix_scoring(main_image)
    
    print(f"\n🎯 BEST CANDIDATE WITH NEW SCORING:")
    print(f"Position: ({best_candidate['x']}, {best_candidate['y']})")
    print(f"New Score: {best_candidate['new_score']:.1f}")
    print(f"Window Size: {best_candidate['window_size']}")
