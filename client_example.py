#!/usr/bin/env python3
"""
Captcha Puzzle Solver - Client Example
Ví dụ sử dụng API để xử lý captcha
"""

import requests
import base64
import json
import time
import sys

def test_api_health(base_url="http://localhost:5000"):
    """Test health check endpoint"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API server đang hoạt động")
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Không thể kết nối API server: {e}")
        return False

def solve_single_captcha(image_path, base_url="http://localhost:5000"):
    """Xử lý 1 captcha"""
    try:
        print(f"🔍 Xử lý captcha: {image_path}")
        
        # Đọc ảnh và convert sang base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
        
        print(f"📦 Base64 length: {len(base64_string)}")
        
        # Gửi request
        start_time = time.time()
        response = requests.post(f"{base_url}/solve-captcha", 
                               json={'image': base64_string},
                               timeout=30)
        processing_time = time.time() - start_time
        
        print(f"⏱️ Thời gian xử lý: {processing_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                coords = result['coordinates']
                print(f"🎯 Thành công!")
                print(f"📍 Tọa độ: ({coords['x']}, {coords['y']})")
                print(f"📍 Raw tọa độ: ({coords['raw_x']}, {coords['raw_y']})")
                print(f"💬 Message: {result['message']}")
                return coords
            else:
                print(f"❌ Thất bại: {result['error']}")
                return None
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return None
            
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {image_path}")
        return None
    except requests.exceptions.Timeout:
        print("❌ Request timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def solve_batch_captcha(image_paths, base_url="http://localhost:5000"):
    """Xử lý nhiều captcha cùng lúc"""
    try:
        print(f"🔍 Xử lý batch {len(image_paths)} captcha")
        
        # Đọc tất cả ảnh và convert sang base64
        base64_images = []
        for image_path in image_paths:
            try:
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                    base64_string = base64.b64encode(image_data).decode('utf-8')
                    base64_images.append(base64_string)
                print(f"✅ Loaded: {image_path}")
            except FileNotFoundError:
                print(f"❌ Không tìm thấy: {image_path}")
                base64_images.append("")  # Placeholder
        
        # Gửi batch request
        start_time = time.time()
        response = requests.post(f"{base_url}/solve-captcha-batch", 
                               json={'images': base64_images},
                               timeout=60)
        processing_time = time.time() - start_time
        
        print(f"⏱️ Thời gian xử lý batch: {processing_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print(f"📊 Kết quả batch:")
            print(f"  - Tổng: {result['total']}")
            print(f"  - Thành công: {result['success_count']}")
            print(f"  - Thất bại: {result['failed_count']}")
            
            print(f"\n📋 Chi tiết:")
            for i, item_result in enumerate(result['results']):
                image_name = image_paths[i] if i < len(image_paths) else f"image_{i}"
                if item_result['success']:
                    coords = item_result['coordinates']
                    print(f"  ✅ {image_name}: ({coords['x']}, {coords['y']})")
                else:
                    print(f"  ❌ {image_name}: {item_result['error']}")
            
            return result
        else:
            print(f"❌ HTTP Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Batch request timeout")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def main():
    """Main function"""
    print("🔧 Captcha Puzzle Solver - Client Test")
    
    base_url = "http://localhost:5000"
    
    # Test health check
    if not test_api_health(base_url):
        print("💡 Hint: Chạy API server trước:")
        print("  python resolveCaptcha.py api")
        return
    
    print("\n" + "="*50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'single':
            # Test single image
            image_path = sys.argv[2] if len(sys.argv) > 2 else 'image.png'
            solve_single_captcha(image_path, base_url)
            
        elif sys.argv[1] == 'batch':
            # Test batch images
            image_paths = sys.argv[2:] if len(sys.argv) > 2 else ['image.png']
            solve_batch_captcha(image_paths, base_url)
            
        else:
            print("❌ Lệnh không hợp lệ!")
            print("Sử dụng:")
            print("  python client_example.py single [image_path]")
            print("  python client_example.py batch [image1] [image2] ...")
    else:
        # Default: test single image
        solve_single_captcha('image.png', base_url)

if __name__ == "__main__":
    main()
