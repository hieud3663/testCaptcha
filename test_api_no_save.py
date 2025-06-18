#!/usr/bin/env python3
"""
Test API không lưu ảnh để tiết kiệm bộ nhớ
"""

import base64
import os
import sys
sys.path.append('/workspaces/testCaptcha')

from resolveCaptcha import process_captcha_image

def test_api_no_save():
    """Test API không lưu ảnh"""
    
    # Đọc ảnh và convert thành base64
    image_path = "image.png"
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
    
    print("🧪 TEST API KHÔNG LUU ẢNH...")
    print(f"📷 File ảnh: {image_path}")
    print(f"📦 Base64 length: {len(base64_string)}")
    
    # Đếm số file trong results trước khi test
    results_dir = "results"
    files_before = []
    if os.path.exists(results_dir):
        files_before = os.listdir(results_dir)
    
    print(f"📁 Số file trong results trước: {len(files_before)}")
    
    # Call API function
    result = process_captcha_image(base64_string)
    
    # Đếm số file trong results sau khi test
    files_after = []
    if os.path.exists(results_dir):
        files_after = os.listdir(results_dir)
    
    print(f"📁 Số file trong results sau: {len(files_after)}")
    
    # Kiểm tra kết quả
    if len(files_after) > len(files_before):
        print("❌ API vẫn đang lưu ảnh! Cần kiểm tra lại.")
        new_files = set(files_after) - set(files_before)
        print(f"📸 File mới được tạo: {new_files}")
    else:
        print("✅ API KHÔNG lưu ảnh - Tiết kiệm bộ nhớ thành công!")
    
    print(f"\n🔍 Kết quả API:")
    print(f"Success: {result.get('success', False)}")
    if result.get('success'):
        coords = result.get('coordinates', {})
        print(f"Coordinates: ({coords.get('x')}, {coords.get('y')})")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    test_api_no_save()
