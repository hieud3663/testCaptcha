from PIL import Image, ImageDraw, ImageFont
import numpy as np
from skimage import feature, filters, morphology, measure
import cv2
from scipy import ndimage
import os
from datetime import datetime

def find_real_gap_position(main_image_path):
    """
    Tìm vị trí gap thực tế trong ảnh - không dùng target cố định
    """
    print("=== TÌM VỊ TRÍ GAP THỰC TẾ ===")
    
    img = Image.open(main_image_path).convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    print(f"📏 Image size: {w}x{h}")
    
    # Tìm tất cả ô vuông gap trong toàn bộ ảnh
    all_gaps = []
    
    # Quét toàn bộ ảnh để tìm các vùng có thể là gap
    window_sizes = [16, 20, 24, 28, 32]
    
    print("🔍 Scanning toàn bộ ảnh để tìm gap...")
    
    # Lấy thống kê background
    bg_samples = []
    for i in range(0, w-50, 50):
        for j in range(0, h-50, 50):
            sample = gray[j:j+20, i:i+20]
            if sample.size > 0:
                bg_samples.append(np.mean(sample))
    
    bg_mean = np.mean(bg_samples) if bg_samples else 128
    print(f"📊 Background mean: {bg_mean:.1f}")
    
    # Quét từng vị trí trong ảnh (tối ưu hóa)
    for window_size in window_sizes:
        half_size = window_size // 2
        
        for y in range(half_size, h - half_size, 8):  # Step = 8px để tăng tốc
            for x in range(half_size, w - half_size, 8):
                
                # Extract window
                window = gray[y - half_size:y + half_size,
                             x - half_size:x + half_size]
                
                if window.shape[0] != window_size or window.shape[1] != window_size:
                    continue
                
                # Phân tích xem có phải gap không
                gap_score = analyze_potential_gap(window, x, y, bg_mean, window_size)
                
                if gap_score > 70:  # Threshold để xác định gap
                    all_gaps.append({
                        'x': x,
                        'y': y,
                        'score': gap_score,
                        'window_size': window_size,
                        'area': window_size * window_size
                    })
    
    print(f"✅ Found {len(all_gaps)} potential gaps")
    
    if not all_gaps:
        print("❌ Không tìm thấy gap nào")
        return []
    
    # Sắp xếp theo score
    all_gaps.sort(key=lambda x: x['score'], reverse=True)
    
    # Loại bỏ duplicates (gaps gần nhau)
    unique_gaps = []
    for gap in all_gaps:
        is_duplicate = False
        for existing in unique_gaps:
            distance = np.sqrt((gap['x'] - existing['x'])**2 + (gap['y'] - existing['y'])**2)
            if distance < 20:  # Nếu gần nhau < 20px thì coi là duplicate
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_gaps.append(gap)
    
    print(f"🎯 Found {len(unique_gaps)} unique gaps:")
    for i, gap in enumerate(unique_gaps[:10]):
        print(f"  {i+1}. ({gap['x']}, {gap['y']}) - Score: {gap['score']:.1f} - Size: {gap['window_size']}")
    
    return unique_gaps

def analyze_potential_gap(window, x, y, bg_mean, window_size):
    """
    Phân tích xem window có phải là gap thực tế không
    """
    window_mean = np.mean(window)
    window_std = np.std(window)
    
    # 1. Gap phải tối hơn background đáng kể
    darkness_ratio = (bg_mean - window_mean) / bg_mean if bg_mean > 0 else 0
    darkness_score = max(0, min(100, darkness_ratio * 300))
    
    # 2. Gap phải có intensity đồng đều (std thấp)
    uniformity_score = max(0, 100 - window_std * 3)
    
    # 3. Gap có hình dạng vuông/chữ nhật
    # Tính variance theo hàng và cột
    row_means = [np.mean(window[i, :]) for i in range(window_size)]
    col_means = [np.mean(window[:, j]) for j in range(window_size)]
    
    row_consistency = max(0, 100 - np.std(row_means) * 8)
    col_consistency = max(0, 100 - np.std(col_means) * 8)
    shape_score = (row_consistency + col_consistency) / 2
    
    # 4. Gap có contrast với viền xung quanh
    border_pixels = np.concatenate([
        window[0, :], window[-1, :],  # Top, Bottom
        window[:, 0], window[:, -1]   # Left, Right
    ])
    border_mean = np.mean(border_pixels)
    
    center_size = max(4, window_size // 2)
    start = (window_size - center_size) // 2
    center_region = window[start:start+center_size, start:start+center_size]
    center_mean = np.mean(center_region)
    
    contrast_score = max(0, min(100, (border_mean - center_mean) / max(border_mean, 1) * 150))
    
    # 5. Kích thước hợp lý
    size_score = 100 if 20 <= window_size <= 30 else 80 if 16 <= window_size <= 34 else 60
    
    # 6. Vị trí không ở rìa ảnh (gaps thường không ở rìa)
    margin = 30
    if x < margin or x > 340-margin or y < margin or y > 251-margin:
        position_penalty = 30
    else:
        position_penalty = 0
    
    # Tổng hợp score
    total_score = (
        darkness_score * 0.25 +      # 25% - Độ tối
        uniformity_score * 0.20 +    # 20% - Đồng đều
        shape_score * 0.20 +         # 20% - Hình dạng vuông
        contrast_score * 0.20 +      # 20% - Contrast
        size_score * 0.15            # 15% - Kích thước
    ) - position_penalty
    
    # Bonus cho gaps có đặc trưng xuất sắc
    if darkness_score > 60 and uniformity_score > 70 and contrast_score > 50:
        total_score *= 1.3
    
    return max(0, total_score)

def visualize_all_gaps(main_image_path, gaps, output_dir="results"):
    """
    Visualize tất cả gaps tìm thấy trên ảnh
    """
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        img = Image.open(main_image_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # Màu sắc khác nhau cho từng gap
        colors = [
            (255, 0, 0),    # Đỏ
            (0, 255, 0),    # Xanh lá  
            (0, 0, 255),    # Xanh dương
            (255, 255, 0),  # Vàng
            (255, 0, 255),  # Tím
            (0, 255, 255),  # Cyan
            (255, 165, 0),  # Cam
            (128, 128, 128) # Xám
        ]
        
        # Vẽ tất cả gaps
        for i, gap in enumerate(gaps[:10]):  # Top 10 gaps
            color = colors[i % len(colors)]
            x, y = gap['x'], gap['y']
            
            # Vẽ circle với size tùy theo score
            radius = max(6, min(15, int(gap['score'] / 10)))
            
            draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                        fill=color, outline=(255, 255, 255), width=2)
            
            # Thêm label số thứ tự
            draw.text((x+radius+2, y-radius), str(i+1), fill=color)
            
            # Thêm info box cho top 3
            if i < 3:
                info_text = f"{i+1}. ({x},{y}) S:{gap['score']:.0f}"
                text_y = 10 + i * 20
                draw.rectangle([10, text_y, 200, text_y+18], 
                              fill=(0, 0, 0), outline=color, width=2)
                draw.text((12, text_y+2), info_text, fill=color)
        
        # Tạo tên file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"all_gaps_{timestamp}.png"
        output_path = os.path.join(output_dir, filename)
        
        img.save(output_path, "PNG", quality=95)
        
        print(f"✅ Đã lưu visualization: {output_path}")
        print("📊 Gaps được đánh dấu:")
        for i, gap in enumerate(gaps[:5]):
            print(f"   {i+1}. ({gap['x']}, {gap['y']}) - Score: {gap['score']:.1f} - {colors[i % len(colors)]}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Lỗi khi lưu visualization: {str(e)}")
        return None

# Sử dụng
if __name__ == "__main__":
    main_image = "image.png"
    
    print("🚀 Tìm vị trí gap thực tế trong ảnh...")
    
    gaps = find_real_gap_position(main_image)
    
    if gaps:
        print(f"\n🎯 Tìm thấy {len(gaps)} gaps!")
        
        # Visualize tất cả gaps
        viz_path = visualize_all_gaps(main_image, gaps)
        
        print(f"\n📋 TOP 5 GAPS TÍNH SCORE CAO NHẤT:")
        for i, gap in enumerate(gaps[:5]):
            print(f"  {i+1}. Vị trí: ({gap['x']}, {gap['y']})")
            print(f"      Score: {gap['score']:.1f}")
            print(f"      Size: {gap['window_size']}x{gap['window_size']}")
            print(f"      Area: {gap['area']} pixels")
            print()
        
        print("👆 VUI LÒNG CHO TÔI BIẾT GAP NÀO LÀ ĐÚNG!")
        print("   (Số thứ tự của gap đúng từ 1-5)")
        
    else:
        print("❌ Không tìm thấy gap nào!")
