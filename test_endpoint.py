"""
RunPod Serverless Test Script for NAFAA Charity Processing

Tests both endpoints:
- full_pipeline: Complete charity assessment (voice + images)
- quality_gate: Image quality assessment only
"""

import requests
import json
import time
import base64
import os

# ⚠️ PASTE YOUR RUNPOD CREDENTIALS HERE
API_KEY = os.environ.get("RUNPOD_API_KEY", "your_api_key_here")  # Get from RunPod dashboard
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "your_endpoint_id_here")  # Get from RunPod dashboard

RUN_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run"
STATUS_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Local test files
SAMPLE_INPUT_PATH = "data/sample_input_voice.json"  # MUST be voice-type
VOICE_PATH = "data/v3.mp3"
IMAGE_PATHS = ["img1.jpg", "img2.jpg"]


def encode_file_to_base64(file_path):
    """Convert file to base64 string"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_json(path):
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def submit_job(payload):
    """Submit job to RunPod"""
    print(f"\n📤 Submitting request to {RUN_URL}")
    r = requests.post(RUN_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    job = r.json()
    job_id = job.get("id")
    if not job_id:
        raise RuntimeError(f"No job id returned: {job}")
    print(f"✅ Job submitted: {job_id}")
    return job_id


def poll_job(job_id, poll_interval=2, timeout_s=300):
    """Poll RunPod for job completion"""
    print(f"\n⏳ Polling job {job_id}...")
    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError("Timed out waiting for RunPod job completion.")

        st = requests.get(f"{STATUS_URL}/{job_id}", headers=headers, timeout=60)
        st.raise_for_status()
        data = st.json()
        status = data.get("status")
        
        print(f"  Status: {status}")

        if status == "COMPLETED":
            print(f"✅ Job completed!")
            return data

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"Job failed with status={status}: {json.dumps(data, indent=2)}")

        time.sleep(poll_interval)


def test_full_pipeline():
    """Test full pipeline endpoint"""
    print("\n" + "="*60)
    print("🔄 TEST 1: FULL PIPELINE ENDPOINT")
    print("="*60)
    
    try:
        # Load request data
        request_data = load_json(SAMPLE_INPUT_PATH)
        
        # Encode voice as base64 (uncomment if needed)
        voice_base64 = f"data:audio/mpeg;base64,{encode_file_to_base64(VOICE_PATH)}"
        
        # Encode images as base64 (uncomment if needed)
        images_base64 = []
        for img_path in IMAGE_PATHS:
            if os.path.exists(img_path):
                images_base64.append(f"data:image/jpeg;base64,{encode_file_to_base64(img_path)}")
        
        payload = {
            "input": {
                "endpoint": "full_pipeline",
                "request_data": request_data,
                "voice_base64": voice_base64,      # Uncomment to use base64 audio
                "images_base64": images_base64,    # Uncomment to use base64 images
            }
        }
        
        job_id = submit_job(payload)
        result = poll_job(job_id)
        
        # Extract output
        output = result.get("output", {})
        
        # Display results
        print("\n--- RESPONSE ---")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        
        # Check for errors
        if output.get("statusCode") != 200:
            print(f"\n❌ Endpoint returned error: {output.get('error', 'Unknown error')}")
            if "trace" in output:
                print(f"\nStack trace:\n{output['trace']}")
            return False
        
        report = output.get("report", {})
        if not report:
            print("\n⚠️  No report in response")
            return False
            
        print("\n--- CHARITY ASSESSMENT REPORT ---")
        print(f"Request ID: {output.get('request_id')}")
        print(f"Decision: {report.get('decision_recommendation', 'N/A')}")
        print(f"Medical Needs: {report.get('medical_needs_analysis', 'N/A')}")
        
        # Safely access cost estimate
        cost_breakdown = report.get('cost_breakdown', {})
        if isinstance(cost_breakdown, dict):
            cost = cost_breakdown.get('total_cost_estimate', 'N/A')
        else:
            cost = 'N/A'
        print(f"Cost Estimate: {cost}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ [ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_quality_gate():
    """Test quality gate endpoint"""
    print("\n" + "="*60)
    print("🖼️  TEST 2: QUALITY GATE ENDPOINT")
    print("="*60)
    
    try:
        # Check if test image exists
        test_image = "data\img1.jpg"  # Adjust path as needed
        if not os.path.exists(test_image):
            print(f"⚠️  Test image not found: {test_image}")
            print("   Skipping quality gate test")
            return False
        
        # Encode image
        image_base64 = encode_file_to_base64(test_image)
        
        payload = {
            "input": {
                "endpoint": "quality_gate",
                "image_base64": f"data:image/jpeg;base64,{image_base64}"
            }
        }
        
        job_id = submit_job(payload)
        result = poll_job(job_id)
        
        # Extract output
        output = result.get("output", {})
        
        # Display results
        print("\n--- RESPONSE ---")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        
        if output.get("statusCode") == 200:
            scores = output.get("quality_scores", {})
            print("\n--- QUALITY ASSESSMENT ---")
            print(f"Quality Score:  {scores.get('quality_score', 'N/A'):.2f}")
            print(f"Blur Score:     {scores.get('blur_score', 'N/A'):.2f}")
            print(f"Lighting Score: {scores.get('lighting_score', 'N/A'):.2f}")
        
        return True
    
    except Exception as e:
        print(f"\n❌ [ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n🚀 NAFAA CHARITY PROCESSING - RUNPOD ENDPOINT TEST")
    print("="*60)
    
    # Check credentials
    if "YOUR_API_KEY" in API_KEY or "your_endpoint" in ENDPOINT_ID:
        print("\n❌ ERROR: Please configure API_KEY and ENDPOINT_ID at top of this file")
        print("   Get them from: https://www.runpod.io/console/serverless")
        return
    
    print(f"API Endpoint: {RUN_URL}")
    print(f"Status URL:  {STATUS_URL}")
    
    # Run tests
    results = []
    
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("Quality Gate", test_quality_gate()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "⚠️  SKIPPED/FAILED"
        print(f"{test_name:30} {status}")


if __name__ == "__main__":
    main()
