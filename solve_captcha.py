#!/usr/bin/env python3
"""
GIẢI CAPTCHA PUZZLE - TÌM VỊ TRÍ GHÉP MẢNH
Sử dụng nhận diện hình ảnh để tìm vị trí đúng cần ghép mảnh phía dưới vào phía trên
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def solve_captcha_puzzle(image_path):
    """
    Giải captcha puzzle bằng cách:
    1. Tách mảnh ghép ở phía dưới
    2. Tìm vị trí khớp ở phía trên bằng template matching
    3. Trả về tọa độ chính xác cần click
    """
    print("🎯 BẮT ĐẦU GIẢI CAPTCHA PUZZLE...")
    
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    print(f"📐 Kích thước ảnh: {w}x{h}")
    
    # Bước 1: Tách mảnh ghép ở phía dưới
    piece_region = extract_puzzle_piece(img_rgb, gray)
    if piece_region is None:
        print("❌ Không tìm thấy mảnh ghép!")
        return None
    
    piece_img, piece_mask, piece_bbox = piece_region
    print(f"🧩 Đã tách mảnh ghép: {piece_bbox}")
    
    # Bước 2: Tìm vị trí khớp ở phía trên
    gap_position = find_matching_position(img_rgb, gray, piece_img, piece_mask, piece_bbox)
    
    if gap_position is None:
        print("❌ Không tìm thấy vị trí khớp!")
        return None
    
    # Bước 3: Visualize kết quả
    visualize_result(img_rgb, piece_bbox, gap_position, piece_img)
    
    print(f"✅ THÀNH CÔNG! Vị trí cần click: {gap_position}")
    return gap_position

def extract_puzzle_piece(img_rgb, gray):
    """Tách mảnh ghép từ phía dưới ảnh"""
    h, w = gray.shape
    
    # Tìm vùng mảnh ghép ở 1/4 dưới của ảnh
    bottom_region = gray[int(h*0.75):, :]
    
    # Tìm contour của mảnh ghép
    # Thử nhiều threshold để tìm mảnh ghép
    for thresh_val in [100, 80, 120, 60, 140]:
        _, binary = cv2.threshold(bottom_region, thresh_val, 255, cv2.THRESH_BINARY)
        
        # Morphology để làm sạch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Mảnh ghép phải có kích thước hợp lý
            if 500 < area < 5000:
                x, y, cw, ch = cv2.boundingRect(contour)
                
                # Điều chỉnh tọa độ về ảnh gốc
                y += int(h*0.75)
                
                # Tạo mask cho mảnh ghép
                mask = np.zeros((ch, cw), dtype=np.uint8)
                contour_shifted = contour - [x, 0]  # Shift contour về origin
                cv2.fillPoly(mask, [contour_shifted], 255)
                
                # Extract mảnh ghép
                piece = img_rgb[y:y+ch, x:x+cw].copy()
                
                print(f"🎯 Tìm thấy mảnh ghép với threshold {thresh_val}")
                print(f"📍 Vị trí: ({x}, {y}), Kích thước: {cw}x{ch}, Diện tích: {area}")
                
                return piece, mask, (x, y, cw, ch)
    
    print("⚠️ Không tìm thấy mảnh ghép phù hợp")
    return None

def find_matching_position(img_rgb, gray, piece_img, piece_mask, piece_bbox):
    """Tìm vị trí khớp cho mảnh ghép ở phía trên"""
    h, w = gray.shape
    px, py, pw, ph = piece_bbox
    
    # Vùng tìm kiếm: phía trên, trừ vùng mảnh ghép
    search_region = gray[:int(h*0.75), :]
    search_h, search_w = search_region.shape
    
    print(f"🔍 Tìm kiếm trong vùng: {search_w}x{search_h}")
    
    # Chuẩn bị template từ mảnh ghép
    piece_gray = cv2.cvtColor(piece_img, cv2.COLOR_RGB2GRAY)
    
    # Template matching với multiple scales
    best_match = None
    best_score = 0
    best_method = None
    
    methods = [
        ('TM_CCOEFF_NORMED', cv2.TM_CCOEFF_NORMED),
        ('TM_CCORR_NORMED', cv2.TM_CCORR_NORMED),
        ('TM_SQDIFF_NORMED', cv2.TM_SQDIFF_NORMED)
    ]
    
    # Thử các scale khác nhau
    scales = [1.0, 0.9, 1.1, 0.8, 1.2]
    
    for scale in scales:
        if scale != 1.0:
            new_w = int(pw * scale)
            new_h = int(ph * scale)
            template = cv2.resize(piece_gray, (new_w, new_h))
            template_mask = cv2.resize(piece_mask, (new_w, new_h))
        else:
            template = piece_gray.copy()
            template_mask = piece_mask.copy()
        
        for method_name, method in methods:
            try:
                # Template matching
                if method == cv2.TM_SQDIFF_NORMED:
                    result = cv2.matchTemplate(search_region, template, method, mask=template_mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    match_val = 1 - min_val  # Đảo ngược cho SQDIFF
                    match_loc = min_loc
                else:
                    result = cv2.matchTemplate(search_region, template, method, mask=template_mask)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    match_val = max_val
                    match_loc = max_loc
                
                # Kiểm tra chất lượng match
                if match_val > best_score and match_val > 0.3:  # Ngưỡng tối thiểu
                    # Xác thực thêm bằng cách kiểm tra vùng xung quanh
                    gx, gy = match_loc
                    tw, th = template.shape[::-1]
                    
                    # Đảm bảo không vượt biên
                    if gx + tw <= search_w and gy + th <= search_h:
                        # Kiểm tra vùng gap có phù hợp không
                        gap_region = search_region[gy:gy+th, gx:gx+tw]
                        gap_quality = evaluate_gap_quality(gap_region, template, template_mask)
                        
                        total_score = match_val * 0.7 + gap_quality * 0.3
                        
                        if total_score > best_score:
                            best_score = total_score
                            best_match = (gx + tw//2, gy + th//2)  # Center của match
                            best_method = f"{method_name} (scale={scale})"
                            
                            print(f"🎯 Match tốt hơn: {best_match}, Score: {total_score:.3f}, Method: {best_method}")
            
            except Exception as e:
                continue
    
    if best_match:
        print(f"🏆 Best match: {best_match} với score {best_score:.3f} ({best_method})")
        return best_match
    
    return None

def evaluate_gap_quality(gap_region, template, template_mask):
    """Đánh giá chất lượng vùng gap"""
    if gap_region.shape != template.shape:
        return 0
    
    # Chỉ xét vùng có mask
    masked_gap = gap_region[template_mask > 0]
    masked_template = template[template_mask > 0]
    
    if len(masked_gap) == 0:
        return 0
    
    # Các tiêu chí đánh giá:
    # 1. Độ tương đồng histogram
    gap_hist = cv2.calcHist([masked_gap], [0], None, [256], [0, 256])
    template_hist = cv2.calcHist([masked_template], [0], None, [256], [0, 256])
    hist_corr = cv2.compareHist(gap_hist, template_hist, cv2.HISTCMP_CORREL)
    
    # 2. Độ tương đồng gradient
    gap_grad = cv2.Sobel(gap_region, cv2.CV_64F, 1, 1, ksize=3)
    template_grad = cv2.Sobel(template, cv2.CV_64F, 1, 1, ksize=3)
    grad_corr = np.corrcoef(gap_grad.flatten(), template_grad.flatten())[0, 1]
    grad_corr = max(0, grad_corr) if not np.isnan(grad_corr) else 0
    
    # 3. Sự khác biệt cường độ
    intensity_diff = np.abs(np.mean(masked_gap) - np.mean(masked_template))
    intensity_score = max(0, 1 - intensity_diff / 255)
    
    # Tổng hợp
    quality = (hist_corr * 0.4 + grad_corr * 0.4 + intensity_score * 0.2)
    return max(0, quality)

def visualize_result(img_rgb, piece_bbox, gap_position, piece_img):
    """Hiển thị kết quả"""
    result_img = img_rgb.copy()
    px, py, pw, ph = piece_bbox
    gx, gy = gap_position
    
    # Vẽ bounding box cho mảnh ghép
    cv2.rectangle(result_img, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
    cv2.putText(result_img, 'PIECE', (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Vẽ vị trí gap
    cv2.circle(result_img, gap_position, 10, (0, 255, 0), -1)
    cv2.circle(result_img, gap_position, 15, (0, 255, 0), 2)
    cv2.putText(result_img, f'GAP ({gx},{gy})', (gx + 20, gy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Vẽ đường nối
    piece_center = (px + pw//2, py + ph//2)
    cv2.arrowedLine(result_img, piece_center, gap_position, (255, 255, 0), 2)
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = f"results/captcha_solved_{timestamp}.png"
    
    plt.figure(figsize=(12, 8))
    plt.imshow(result_img)
    plt.title(f'CAPTCHA SOLVED - Gap position: {gap_position}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"💾 Đã lưu kết quả: {result_path}")

def main():
    """Hàm main để chạy chương trình"""
    image_path = "image.png"  # Ảnh captcha
    
    try:
        gap_position = solve_captcha_puzzle(image_path)
        if gap_position:
            print(f"\n🎉 HOÀN THÀNH!")
            print(f"📍 Vị trí cần click: {gap_position}")
            print(f"💡 Hướng dẫn: Click vào tọa độ ({gap_position[0]}, {gap_position[1]})")
        else:
            print("\n❌ THẤT BẠI - Không tìm thấy vị trí phù hợp")
    
    except Exception as e:
        print(f"❌ LỖI: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
