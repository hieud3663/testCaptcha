from PIL import Image, ImageDraw
import numpy as np
import os
from datetime import datetime

def find_gaps_simple():
    """Tìm gaps đơn giản"""
    print("=== TÌM GAP ĐỞN GIẢN ===")
    
    img = Image.open("image.png").convert('RGB')
    img_array = np.array(img)
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    
    h, w = gray.shape
    print(f"Image size: {w}x{h}")
    
    # Tìm vùng tối (có thể là gaps)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    print(f"Mean: {mean_intensity:.1f}, Std: {std_intensity:.1f}")
    
    # Threshold để tìm vùng tối
    dark_threshold = mean_intensity - std_intensity
    dark_areas = gray < dark_threshold
    
    print(f"Dark threshold: {dark_threshold:.1f}")
    print(f"Dark pixels: {np.sum(dark_areas)}")
    
    # Tìm connected components
    from skimage import measure
    labeled = measure.label(dark_areas)
    regions = measure.regionprops(labeled)
    
    print(f"Found {len(regions)} dark regions")
    
    # Filter regions by size and shape
    gaps = []
    for i, region in enumerate(regions):
        area = region.area
        bbox = region.bbox
        height = bbox[2] - bbox[0]
        width = bbox[3] - bbox[1]
        
        # Filter: area 100-2000, aspect ratio gần 1
        if 100 < area < 2000 and 0.5 < width/height < 2.0:
            centroid_y, centroid_x = region.centroid
            gaps.append({
                'x': int(centroid_x),
                'y': int(centroid_y),
                'area': area,
                'width': width,
                'height': height,
                'aspect_ratio': width/height
            })
    
    print(f"Found {len(gaps)} potential gaps")
    
    # Sort by area (gaps thường có area vừa phải)
    gaps.sort(key=lambda x: abs(x['area'] - 500))
    
    # Visualize
    draw_img = img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for i, gap in enumerate(gaps[:5]):
        color = colors[i]
        x, y = gap['x'], gap['y']
        
        # Vẽ circle
        radius = 10
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    outline=color, width=3)
        
        # Vẽ text
        draw.text((x+15, y-10), f"{i+1}", fill=color)
        
        print(f"Gap {i+1}: ({x}, {y}) - Area: {gap['area']}, Size: {gap['width']:.0f}x{gap['height']:.0f}")
    
    # Save
    output_path = f"results/gaps_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    if not os.path.exists("results"):
        os.makedirs("results")
    draw_img.save(output_path)
    
    print(f"Saved: {output_path}")
    return gaps

if __name__ == "__main__":
    gaps = find_gaps_simple()
