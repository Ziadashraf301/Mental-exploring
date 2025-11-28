"""
Full API Testing Script
Covers all endpoints: root, health, users, emotion, depression, sentiment, analytics
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def print_section(title):
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_result(endpoint, status, data):
    status_emoji = "✓" if status < 400 else "✗"
    print(f"{status_emoji} {endpoint} - Status: {status}")
    if isinstance(data, dict):
        print(f"   Response: {json.dumps(data, indent=2)[:200]}...")
    print()

# ============================
# ROOT & HEALTH
# ============================
print_section("Root & Health Endpoints")
for ep in ["/", "/health"]:
    r = requests.get(f"{BASE_URL}{ep}")
    print_result(f"GET {ep}", r.status_code, r.json())

# ============================
# USER MANAGEMENT
# ============================
print_section("User Management")
user_data = {"name": "Test User", "email": "test@example.com", "metadata": {"source": "test_script"}}
r = requests.post(f"{BASE_URL}/users", json=user_data)
print_result("POST /users", r.status_code, r.json())
user_id = r.json().get("user_id") if r.status_code < 400 else None

if user_id:
    for ep in [f"/users/{user_id}", f"/users/{user_id}/stats", f"/users/{user_id}/predictions"]:
        r = requests.get(f"{BASE_URL}{ep}")
        print_result(f"GET {ep}", r.status_code, r.json())
    # Update user
    r = requests.patch(f"{BASE_URL}/users/{user_id}", params={"name": "Updated Test User"})
    print_result(f"PATCH /users/{user_id}", r.status_code, r.json())

# ============================
# EMOTION DETECTION
# ============================
print_section("Emotion Detection")
for ep in ["/emotion/health", "/emotion/model/info"]:
    r = requests.get(f"{BASE_URL}{ep}")
    print_result(f"GET {ep}", r.status_code, r.json())

# Test emotion prediction
image_path = "assets/basket.jpg"
if Path(image_path).exists():
    with open(image_path, "rb") as f:
        files = {"file": f}
        params = {"user_id": user_id} if user_id else {}
        r = requests.post(f"{BASE_URL}/emotion/predict", files=files, params=params)
        print_result("POST /emotion/predict", r.status_code, r.json())
else:
    print(f"⚠️  Skipping emotion prediction - image not found: {image_path}")

# ============================
# DEPRESSION DETECTION
# ============================
print_section("Depression Detection")
dep_payload = {"text": "I feel very sad today.", "user_id": user_id}
for ep in ["/depression/health", "/depression/model/info"]:
    r = requests.get(f"{BASE_URL}{ep}")
    print_result(f"GET {ep}", r.status_code, r.json())

r = requests.post(f"{BASE_URL}/depression/predict", json=dep_payload)
print_result("POST /depression/predict", r.status_code, r.json())

# ============================
# SENTIMENT ANALYSIS
# ============================
print_section("Sentiment Analysis")
sent_payload = {"text": "I love using this API!", "user_id": user_id}
for ep in ["/sentiment/health", "/sentiment/model/info"]:
    r = requests.get(f"{BASE_URL}{ep}")
    print_result(f"GET {ep}", r.status_code, r.json())

r = requests.post(f"{BASE_URL}/sentiment/predict", json=sent_payload)
print_result("POST /sentiment/predict", r.status_code, r.json())

# ============================
# ANALYTICS
# ============================
print_section("Analytics Endpoints")
analytics_endpoints = [
    "/analytics",
    "/analytics/realtime",
    "/analytics/summary",
    "/analytics/trends",
    "/analytics/service/emotion",
    "/analytics/performance",
    "/analytics/predictions/distribution",
    "/analytics/export"
]

for ep in analytics_endpoints:
    params = {"days": 30} if "days" in ep or "trends" in ep else {}
    if "export" in ep:
        params.update({"format": "json"})
    r = requests.get(f"{BASE_URL}{ep}", params=params)
    print_result(f"GET {ep}", r.status_code, r.json())

# ============================
# CLEANUP
# ============================
print_section("Cleanup")
if user_id:
    r = requests.delete(f"{BASE_URL}/users/{user_id}")
    print_result(f"DELETE /users/{user_id}", r.status_code, r.json())

print("\n" + "="*70)
print("  All API Tests Complete!")
print("="*70 + "\n")
