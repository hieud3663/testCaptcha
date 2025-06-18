#!/usr/bin/env python3

from PIL import Image
import numpy as np

def check_gap_result():
    """
    Kiểm tra xem vị trí gap đã tìm được có chính xác không
    """
    print("=== KIỂM TRA KẾT QUẢ GAP ===")
    
    # Đọc ảnh gốc
    img = Image.open('image.png').convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    print(f"📏 Image size: {img.size}")
    print(f"📐 Array shape: {img_array.shape}")
    
    # Vị trí gap đã tìm được
    gap_x, gap_y = 239, 138
    print(f"🎯 Gap position found: ({gap_x}, {gap_y})")
    
    # Phân tích vùng gap
    window_size = 24
    half_size = window_size // 2
    
    if (gap_x - half_size >= 0 and gap_x + half_size < img.width and 
        gap_y - half_size >= 0 and gap_y + half_size < img.height):
        
        gap_window = gray[gap_y - half_size:gap_y + half_size, 
                         gap_x - half_size:gap_x + half_size]
        
        print(f"\n🔍 Gap window analysis:")
        print(f"  Mean: {np.mean(gap_window):.1f}")
        print(f"  Std: {np.std(gap_window):.1f}")
        print(f"  Min: {np.min(gap_window):.1f}")
        print(f"  Max: {np.max(gap_window):.1f}")
        
        # So sánh với background patterns
        bg_samples = [
            gray[:50, :50],           # Top-left
            gray[:50, -50:],          # Top-right
            gray[-50:, :50],          # Bottom-left (có thể là piece)
            gray[-50:, -50:]          # Bottom-right (có thể là piece)
        ]
        
        print(f"\n📊 Background comparison:")
        for i, bg in enumerate(bg_samples):
            bg_mean = np.mean(bg)
            print(f"  BG{i+1} mean: {bg_mean:.1f}")
            
            if np.mean(gap_window) < bg_mean - 30:
                print(f"    ✅ Gap is significantly darker than BG{i+1}")
            elif np.mean(gap_window) < bg_mean - 10:
                print(f"    ⚠️ Gap is somewhat darker than BG{i+1}")
            else:
                print(f"    ❌ Gap is not darker than BG{i+1}")
        
        # Kiểm tra tính đồng đều (uniformity)
        if np.std(gap_window) < 20:
            print(f"✅ Gap window is uniform (std={np.std(gap_window):.1f})")
        else:
            print(f"❌ Gap window is not uniform (std={np.std(gap_window):.1f})")
        
        # Kiểm tra có phải là vùng rỗng/gap thực sự không
        if np.mean(gap_window) < 50:  # Rất tối
            print(f"✅ Gap region is very dark - likely a real gap!")
        elif np.mean(gap_window) < 100:  # Tối vừa
            print(f"⚠️ Gap region is moderately dark - could be a gap")
        else:
            print(f"❌ Gap region is too bright - probably not a gap")
            
    else:
        print("❌ Gap position is out of bounds!")
        return False
    
    # So sánh với các vị trí khác để validation
    print(f"\n🔍 Comparison with other positions:")
    test_positions = [
        (50, 50),    # Top area
        (100, 100),  # Middle area
        (200, 50),   # Top-right area
        (50, 100),   # Left area
    ]
    
    for i, (test_x, test_y) in enumerate(test_positions):
        if (test_x - half_size >= 0 and test_x + half_size < img.width and 
            test_y - half_size >= 0 and test_y + half_size < img.height):
            
            test_window = gray[test_y - half_size:test_y + half_size, 
                              test_x - half_size:test_x + half_size]
            test_mean = np.mean(test_window)
            
            print(f"  Position ({test_x}, {test_y}): mean={test_mean:.1f}")
            
            if np.mean(gap_window) < test_mean - 20:
                print(f"    ✅ Gap is darker than this position")
            else:
                print(f"    ❌ Gap is not significantly darker")
    
    return True

if __name__ == "__main__":
    check_gap_result()
