"""
API Testing Script
Tests all endpoints to verify functionality
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(endpoint, status, data):
    """Print test result"""
    status_emoji = "✓" if status < 400 else "✗"
    print(f"{status_emoji} {endpoint} - Status: {status}")
    if isinstance(data, dict):
        print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
    print()

# ===========================================
# TEST ROOT ENDPOINTS
# ===========================================
print_section("Testing Root Endpoints")

# Test root
response = requests.get(f"{BASE_URL}/")
print_result("GET /", response.status_code, response.json())

# Test health
response = requests.get(f"{BASE_URL}/health")
print_result("GET /health", response.status_code, response.json())

# ===========================================
# TEST USER MANAGEMENT
# ===========================================
print_section("Testing User Management")

# Create user
user_data = {
    "name": "Test User",
    "email": "test@example.com",
    "metadata": {"source": "test_script"}
}
response = requests.post(f"{BASE_URL}/users", json=user_data)
print_result("POST /users", response.status_code, response.json())

user_id = None
if response.status_code == 200:
    user_id = response.json()["user_id"]
    print(f"   Created user_id: {user_id}")

# Get user
if user_id:
    response = requests.get(f"{BASE_URL}/users/{user_id}")
    print_result(f"GET /users/{user_id}", response.status_code, response.json())

# Update user
if user_id:
    response = requests.patch(
        f"{BASE_URL}/users/{user_id}",
        params={"name": "Updated Test User"}
    )
    print_result(f"PATCH /users/{user_id}", response.status_code, response.json())

# Get user stats (will be empty initially)
if user_id:
    response = requests.get(f"{BASE_URL}/users/{user_id}/stats")
    print_result(f"GET /users/{user_id}/stats", response.status_code, response.json())

# Get user predictions (will be empty initially)
if user_id:
    response = requests.get(f"{BASE_URL}/users/{user_id}/predictions")
    print_result(f"GET /users/{user_id}/predictions", response.status_code, response.json())

# ===========================================
# TEST EMOTION DETECTION
# ===========================================
print_section("Testing Emotion Detection")

# Check emotion service health
response = requests.get(f"{BASE_URL}/emotion/health")
print_result("GET /emotion/health", response.status_code, response.json())

# Get model info
response = requests.get(f"{BASE_URL}/emotion/model/info")
print_result("GET /emotion/model/info", response.status_code, response.json())

# Test prediction (requires an image file)
# Note: You need to provide an actual image file path
image_path = "test_image.jpg"  # Change this to your test image
if Path(image_path).exists():
    print(f"\n   Testing with image: {image_path}")
    files = {"file": open(image_path, "rb")}
    params = {"user_id": user_id} if user_id else {}
    
    response = requests.post(
        f"{BASE_URL}/emotion/predict",
        files=files,
        params=params
    )
    print_result("POST /emotion/predict", response.status_code, response.json())
else:
    print(f"   ⚠️  Skipping emotion prediction - image not found: {image_path}")

# ===========================================
# TEST ANALYTICS
# ===========================================
print_section("Testing Analytics")

# Get general analytics
response = requests.get(f"{BASE_URL}/analytics")
print_result("GET /analytics", response.status_code, response.json())

# Get analytics with params
response = requests.get(f"{BASE_URL}/analytics", params={"days": 30})
print_result("GET /analytics?days=30", response.status_code, response.json())

# Get realtime analytics
response = requests.get(f"{BASE_URL}/analytics/realtime")
print_result("GET /analytics/realtime", response.status_code, response.json())

# Get service analytics
response = requests.get(f"{BASE_URL}/analytics/service/emotion", params={"days": 7})
print_result("GET /analytics/service/emotion", response.status_code, response.json())

# Get performance metrics
response = requests.get(f"{BASE_URL}/analytics/performance", params={"days": 7})
print_result("GET /analytics/performance", response.status_code, response.json())

# Get trends
response = requests.get(f"{BASE_URL}/analytics/trends", params={"days": 30})
print_result("GET /analytics/trends", response.status_code, response.json())

# Get summary
response = requests.get(f"{BASE_URL}/analytics/summary")
print_result("GET /analytics/summary", response.status_code, response.json())

# Get distribution
response = requests.get(f"{BASE_URL}/analytics/predictions/distribution", params={"days": 7})
print_result("GET /analytics/predictions/distribution", response.status_code, response.json())

# Export analytics
response = requests.get(f"{BASE_URL}/analytics/export", params={"days": 7, "format": "json"})
print_result("GET /analytics/export", response.status_code, response.json())

# ===========================================
# CLEANUP (OPTIONAL)
# ===========================================
print_section("Cleanup")

# Delete test user
if user_id:
    response = requests.delete(f"{BASE_URL}/users/{user_id}")
    print_result(f"DELETE /users/{user_id}", response.status_code, response.json())

print("\n" + "="*70)
print("  Testing Complete!")
print("="*70 + "\n")