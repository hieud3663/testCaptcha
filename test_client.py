#!/usr/bin/env python3
"""
Client Test đơn giản cho API
"""

import requests
import base64
import json
import os
import time

def test_api_simple():
    """Test API đơn giản"""
    base_url = "http://localhost:8082"
    
    print("🧪 API CLIENT TEST")
    print("=" * 50)
    
    # Test Health
    print("📡 Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health: {response.json()}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return
    
    # Test Solve Captcha
    print("\n🎯 Testing solve captcha...")
    
    # Đọc ảnh
    image_path = "image.png"
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
    
    print(f"📷 Image: {len(image_data)} bytes")
    
    # Gửi request
    payload = {"image": base64_string}
    
    print("🚀 Sending solve request...")
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{base_url}/solve-captcha",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=45  # Tăng timeout
        )
        
        end_time = time.time()
        
        print(f"📤 Status: {response.status_code}")
        print(f"⏱️ Time: {end_time - start_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"❌ Error response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("❌ Request timeout!")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    test_api_simple()
