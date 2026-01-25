"""
RunPod Serverless Handler for NAFAA Charity Request Processing

Accepts JSON input with voice file and images, processes through full pipeline,
returns structured charity assessment report.
"""

import json
import base64
import tempfile
import os
import traceback
import logging
import runpod

# Import main pipeline
from main import process_request, load_json, save_json
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_quality_gate():
    """Dynamically load quality gate module"""
    spec = importlib.util.spec_from_file_location(
        "quality_gate",
        os.path.join(os.path.dirname(__file__), "images checks", "quality_gate_finalized.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def handler(job):
    """
    RunPod Serverless entry point with routing to different endpoints
    
    Request format:
    {
      "input": {
        "endpoint": "full_pipeline" or "quality_gate",
        "request_data": {...},  // For full_pipeline
        "voice_base64": "...",  // Optional
        "images_base64": [...], // Optional
        "image_base64": "..."   // For quality_gate endpoint
      }
    }
    """
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    endpoint = job_input.get("endpoint", "full_pipeline").lower()
    
    if endpoint == "quality_gate":
        return quality_gate_endpoint(job_input)
    elif endpoint == "full_pipeline":
        return full_pipeline_endpoint(job_input)
    else:
        return _error(f"Unknown endpoint: {endpoint}. Use 'full_pipeline' or 'quality_gate'")


def quality_gate_endpoint(job_input):
    """
    Quality Gate only endpoint
    
    Request:
    {
      "endpoint": "quality_gate",
      "image_base64": "<base64 image>" or "image_path": "/path/to/image.jpg"
    }
    
    Returns:
    {
      "statusCode": 200,
      "quality_scores": {
        "quality_score": 0.0-1.0,
        "blur_score": 0.0-1.0,
        "lighting_score": 0.0-1.0
      }
    }
    """
    temp_files = []
    
    try:
        image_path = None
        
        # Handle base64 image
        if "image_base64" in job_input:
            try:
                img_b64 = job_input["image_base64"]
                if "base64," in img_b64:
                    img_b64 = img_b64.split("base64,", 1)[1]
                elif "," in img_b64 and img_b64.lower().startswith("data:"):
                    img_b64 = img_b64.split(",", 1)[1]
                
                img_bytes = base64.b64decode(img_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp.write(img_bytes)
                    image_path = tmp.name
                    temp_files.append(image_path)
                
                logger.info(f"Image saved to {image_path}")
            except Exception as e:
                return _error(f"Failed to decode image: {str(e)}")
        
        # Or use provided image path
        elif "image_path" in job_input:
            image_path = job_input["image_path"]
            if not os.path.exists(image_path):
                return _error(f"Image not found: {image_path}")
        
        else:
            return _error("No image_base64 or image_path provided")
        
        # Load quality gate module
        quality_gate_module = load_quality_gate()
        
        # Check if check_quality function exists, else use assess_image_quality
        if hasattr(quality_gate_module, 'check_quality'):
            quality_result = quality_gate_module.check_quality(image_path)
        elif hasattr(quality_gate_module, 'assess_image_quality'):
            quality_result = quality_gate_module.assess_image_quality(image_path)
        else:
            return _error("Quality gate module missing check_quality or assess_image_quality function")
        
        logger.info(f"Quality assessment complete: {quality_result}")
        
        return {
            "statusCode": 200,
            "quality_scores": quality_result
        }
    
    except Exception as e:
        logger.error(f"Quality gate error: {str(e)}\n{traceback.format_exc()}")
        return {
            "statusCode": 500,
            "error": str(e),
            "trace": traceback.format_exc()
        }
    
    finally:
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                    logger.info(f"Cleaned up {tmp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {tmp_file}: {e}")


def full_pipeline_endpoint(job_input):
    """
    Full pipeline endpoint - processes complete request through all 8 stages
    
    Expected request:
    {
      "endpoint": "full_pipeline",
      "request_data": {  // Full request JSON matching sample_input_voice.json
        "request_id": "REQ-2026-001",
        "request_category": "Medical Aid",
        "request_description": {
          "type": "voice",
          "voice_path": "v3.mp3"  // or base64 audio below
        },
        "evidence_images": [
          {
            "image_id": "IMG-001",
            "image_path": "img1.jpg",  // or base64 below
            "ocr_extracted_text": "..."
          }
        ]
      },
      "voice_base64": "<base64 encoded audio>",  // Optional: will save to temp file
      "images_base64": ["<base64>", "<base64>", ...]  // Optional: will save to temp files
    }
    """
    temp_files = []
    
    try:
        
        # Get request data
        request_data = job_input.get("request_data", {})
        if not request_data:
            return _error("No request_data provided")
        
        # Handle optional base64 audio
        if "voice_base64" in job_input:
            try:
                voice_b64 = job_input["voice_base64"]
                if "base64," in voice_b64:
                    voice_b64 = voice_b64.split("base64,", 1)[1]
                
                voice_bytes = base64.b64decode(voice_b64)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(voice_bytes)
                    voice_path = tmp.name
                    temp_files.append(voice_path)
                
                # Update request_data to point to temp file
                request_data["request_description"]["voice_path"] = voice_path
                logger.info(f"Voice audio saved to {voice_path}")
            except Exception as e:
                return _error(f"Failed to decode voice audio: {str(e)}")
        
        # Handle optional base64 images
        if "images_base64" in job_input:
            images_b64 = job_input["images_base64"]
            if not isinstance(images_b64, list):
                images_b64 = [images_b64]
            
            evidence_images = request_data.get("evidence_images", [])
            
            for idx, img_b64 in enumerate(images_b64):
                try:
                    if "base64," in img_b64:
                        img_b64 = img_b64.split("base64,", 1)[1]
                    elif "," in img_b64 and img_b64.lower().startswith("data:"):
                        img_b64 = img_b64.split(",", 1)[1]
                    
                    img_bytes = base64.b64decode(img_b64)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                        tmp.write(img_bytes)
                        img_path = tmp.name
                        temp_files.append(img_path)
                    
                    # Update or create evidence image entry
                    if idx < len(evidence_images):
                        evidence_images[idx]["image_path"] = img_path
                    else:
                        evidence_images.append({
                            "image_id": f"IMG-{idx+1:03d}",
                            "image_path": img_path,
                            "ocr_extracted_text": ""
                        })
                    
                    logger.info(f"Image {idx} saved to {img_path}")
                except Exception as e:
                    logger.warning(f"Failed to decode image {idx}: {str(e)}")
            
            request_data["evidence_images"] = evidence_images
        
        # Save request to temp JSON file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_input:
            json.dump(request_data, tmp_input, ensure_ascii=False, indent=2)
            input_path = tmp_input.name
            temp_files.append(input_path)
        
        # Process through main pipeline
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_output:
            output_path = tmp_output.name
            temp_files.append(output_path)
        
        logger.info(f"Processing request {request_data.get('request_id', 'UNKNOWN')}")
        success = process_request(input_path, output_path)
        
        if not success:
            return _error("Pipeline processing failed")
        
        # Load and return results
        report = load_json(output_path)
        
        return {
            "statusCode": 200,
            "request_id": request_data.get("request_id"),
            "report": report
        }
    
    except Exception as e:
        logger.error(f"Handler error: {str(e)}\n{traceback.format_exc()}")
        return {
            "statusCode": 500,
            "error": str(e),
            "trace": traceback.format_exc()
        }
    
    finally:
        # Cleanup temp files
        for tmp_file in temp_files:
            try:
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                    logger.info(f"Cleaned up {tmp_file}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {tmp_file}: {e}")


def _error(msg):
    """Format error response"""
    logger.error(msg)
    return {
        "statusCode": 400,
        "error": msg
    }


# 🔥 REQUIRED for RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler
    })
