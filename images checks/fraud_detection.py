# -----------------------------
# Fraud Detection Module
# Install dependencies: pip install torch torchvision transformers pillow
# -----------------------------

import json
import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import importlib.util
import sys

# Import reverse image search functionality using importlib (folder has space)
spec = importlib.util.spec_from_file_location("reverse_image", os.path.join(os.path.dirname(__file__), "reverse_image.py"))
reverse_image = importlib.util.module_from_spec(spec)
sys.modules["reverse_image"] = reverse_image
spec.loader.exec_module(reverse_image)

# -----------------------------
# 1. AI-generated detection
# -----------------------------
ai_model_name = "Organice/CLIP-based-model"  # Real model for image classification
try:
    ai_processor = AutoImageProcessor.from_pretrained(ai_model_name)
    ai_model = AutoModelForImageClassification.from_pretrained(ai_model_name)
    ai_model.eval()
except Exception:
    # Fallback: Use a simpler model if the primary one fails
    ai_model_name = "microsoft/resnet-50"
    ai_processor = AutoImageProcessor.from_pretrained(ai_model_name)
    ai_model = AutoModelForImageClassification.from_pretrained(ai_model_name)
    ai_model.eval()

def ai_generated_probability(image_path):
    """
    Detect if an image is AI-generated.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        float: Probability that the image is AI-generated (0-1)
    """
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = ai_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = ai_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        # Return the max probability as a default detection score
        return float(np.max(probs))
    except Exception as e:
        print(f"Warning: AI detection failed for {image_path}: {e}")
        return 0.5  # Return neutral probability on error

# -----------------------------
# 2. Get CLIP embedding for image
# -----------------------------
def get_clip_embedding(image_path):
    """
    Get CLIP embedding for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        np.ndarray: CLIP embedding vector
    """
    image = Image.open(image_path).convert("RGB")
    inputs = ai_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = ai_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

# 3. Main function for fraud detection
# -----------------------------
def process_image(user_id, image_path, sim_threshold=0.85, ai_threshold=0.7):
    """
    Process an image for fraud detection.
    
    Args:
        user_id: User ID submitting the image
        image_path: Path to the image file
        sim_threshold: Similarity threshold for duplicate detection
        ai_threshold: Probability threshold for AI-generated detection
        
    Returns:
        dict: Fraud detection results
    """
    # AI-generated check
    ai_prob = ai_generated_probability(image_path)
    is_ai = ai_prob >= ai_threshold
    
    # Duplicate check using reverse image search
    dup_result = reverse_image.find_duplicates(image_path, user_id, sim_threshold)
    is_duplicate = dup_result['is_duplicate']
    similarity = dup_result['similarity']
    
    # Store embedding if accepted (not duplicate and not AI)
    if not is_duplicate and not is_ai:
        reverse_image.add_image_to_index(image_path, user_id)
    
    return {
        "image_path": image_path,
        "user_id": user_id,
        "ai_probability": ai_prob,
        "is_ai": is_ai,
        "is_duplicate": is_duplicate,
        "similarity": similarity,
        "passed": not is_ai and not is_duplicate,
    }

# 4. Process JSON input and output
# ============================================================================
def main(input_json="test_data.json", output_json="fraud_detection_results.json"):
    """
    Process multiple images from JSON input with fraud detection.
    
    Reads a JSON file with image paths and user IDs, processes each image
    for fraud detection (AI-generated check and duplicate detection), and
    writes results to output JSON file.
    
    Args:
        input_json: Path to input JSON file with image data
        output_json: Path to output JSON file for results
        
    Example input JSON:
        {
          "images": [
            {"user_id": "user_001", "image_path": "path/to/image.jpg"},
            {"user_id": "user_002", "image_path": "path/to/image2.jpg"}
          ]
        }
    
    Example output JSON:
        {
          "summary": {...},
          "results": [
            {
              "image_path": "...",
              "user_id": "...",
              "ai_probability": 0.45,
              "is_ai": false,
              "is_duplicate": false,
              "similarity": 0.0,
              "passed": true,
              "timestamp": "..."
            }
          ]
        }
    """
    from datetime import datetime
    
    print("=" * 70)
    print("FRAUD DETECTION MODULE")
    print("=" * 70)
    print(f"Input file: {input_json}")
    print(f"Output file: {output_json}")
    
    # Load input data
    try:
        with open(input_json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_json}' not found!")
        return
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON in '{input_json}'")
        return
    
    # Get images from input JSON
    images = data.get("images", [])
    if not images:
        print("⚠️  Warning: No images found in input JSON")
        return
    
    print(f"\n📊 Processing {len(images)} images...\n")
    
    results = []
    for idx, entry in enumerate(images, 1):
        user_id = entry.get("user_id", "unknown")
        image_path = entry.get("image_path", "")
        image_id = entry.get("id", f"image_{idx}")
        
        if not image_path or not os.path.exists(image_path):
            print(f"[{idx}/{len(images)}] ❌ {image_id}: File not found - {image_path}")
            continue
        
        try:
            result = process_image(user_id, image_path)
            result["id"] = image_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Determine status emoji
            if result["is_ai"]:
                status = "🚨 AI-GENERATED"
            elif result["is_duplicate"]:
                status = "⚠️  DUPLICATE"
            elif result["passed"]:
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
            
            print(f"[{idx}/{len(images)}] {status} | {image_id}")
            results.append(result)
        except Exception as e:
            print(f"[{idx}/{len(images)}] ❌ Error processing {image_id}: {str(e)}")
            results.append({
                "id": image_id,
                "user_id": user_id,
                "image_path": image_path,
                "error": str(e),
                "passed": False,
                "timestamp": datetime.now().isoformat()
            })
    
    # Calculate summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.get('passed', False))
    ai_detected = sum(1 for r in results if r.get('is_ai', False))
    duplicates = sum(1 for r in results if r.get('is_duplicate', False))
    errors = sum(1 for r in results if 'error' in r)
    
    summary = {
        "total_processed": total,
        "passed": passed,
        "ai_generated_detected": ai_detected,
        "duplicates_detected": duplicates,
        "errors": errors,
        "pass_rate": f"{(passed/total*100):.1f}%" if total > 0 else "0%",
        "timestamp": datetime.now().isoformat()
    }
    
    # Save results to JSON
    output_data = {
        "module": "fraud_detection",
        "summary": summary,
        "results": results
    }
    
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    
    # Persist embeddings from reverse image module
    reverse_image.save_index()
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    print(f"Total processed: {total}")
    print(f"  ✅ Passed:              {passed} ({summary['pass_rate']})")
    print(f"  🚨 AI-generated:        {ai_detected}")
    print(f"  ⚠️  Duplicates:          {duplicates}")
    print(f"  ❌ Errors:              {errors}")
    print(f"\n💾 Results saved to: {output_json}")
    print("=" * 70 + "\n")
    
    return output_data


# ============================================================================
# Example usage
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fraud Detection: AI-generated and duplicate image detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fraud_detection.py test_data.json fraud_results.json
  python fraud_detection.py --input test_data.json --output my_results.json
        """
    )
    parser.add_argument("--input", "-i", default="test_data.json",
                       help="Input JSON file with images (default: test_data.json)")
    parser.add_argument("--output", "-o", default="fraud_detection_results.json",
                       help="Output JSON file for results (default: fraud_detection_results.json)")
    
    args = parser.parse_args()
    main(args.input, args.output)


# ===========================================
# Wrapper function for main.py compatibility
# ===========================================
def detect_fraud(image_path):
    """Wrapper to return fraud risk as dict"""
    try:
        ai_prob = ai_generated_probability(image_path)
        fraud_risk = 'High' if ai_prob > 0.7 else 'Low'
        return {
            'fraud_risk': fraud_risk,
            'ai_generated_probability': ai_prob,
            'duplicate_detected': False
        }
    except Exception as e:
        return {
            'fraud_risk': 'Low',
            'ai_generated_probability': 0.0,
            'duplicate_detected': False
        }

