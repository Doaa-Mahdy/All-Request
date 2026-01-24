from typing import List, Dict, Any, Tuple, Optional
import json
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
import os
import base64

# ============================================================================
# GLOBAL MODEL LOADING (LOAD ONCE)
# ============================================================================

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("[VQA] Loading Qwen2-VL model...")

if DEVICE == "cuda":
    QWEN_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
else:
    QWEN_MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True
    )

QWEN_MODEL.tie_weights()
QWEN_PROCESSOR = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

QWEN_MODEL.eval()

print("[VQA] Model loaded successfully.")


def run_vqa(image_path: str, question: str) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    prompt = QWEN_PROCESSOR.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    device = next(QWEN_MODEL.parameters()).device
    inputs = QWEN_PROCESSOR(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.inference_mode():
        output_ids = QWEN_MODEL.generate(**inputs, max_new_tokens=128)

    generated = output_ids[0][inputs.input_ids.shape[1]:]
    answer = QWEN_PROCESSOR.decode(
        generated,
        skip_special_tokens=True
    )

    return answer.strip()


# ============================================================================
# MAIN VQA FUNCTIONS
# ============================================================================

def answer_questions(
    images: List[str],
    category: str,
    vqa_questions: Dict[str, List[str]],
    ocr_texts: Optional[List[str]] = None
) -> Dict[str, Any]:

    selected_questions = select_category_questions(category, vqa_questions)

    results = []
    all_confidences = []

    for idx, image_path in enumerate(images):
        image_results = []

        ocr_text = ocr_texts[idx] if ocr_texts and idx < len(ocr_texts) else ""

        for question in selected_questions:
            qa = answer_single_question(
                image_path=image_path,
                question=question,
                ocr_text=ocr_text,
                context=f"Category: {category}"
            )
            image_results.append(qa)
            all_confidences.append(qa["confidence"])

        consistency_prompt = (
            "النص المذكور:\n"
            f"{ocr_text}\n\n"
            "هل الصورة تتفق مع هذا النص؟\n"
            "أجب بنعم أو لا أو غير واضح مع توضيح مختصر."
        )

        consistency_answer = answer_single_question(
            image_path=image_path,
            question=consistency_prompt,
            ocr_text=ocr_text,
            context=consistency_prompt
        )

        image_results.append({
            **consistency_answer,
            "question_type": "image_text_consistency"
        })

        results.append({
            "image_id": f"IMG-{idx+1:03d}",
            "image_path": image_path,
            "vqa_results": image_results
        })

    overall_confidence = sum(all_confidences) / max(len(all_confidences), 1)

    return {
        "category": category,
        "images": results,
        "overall_confidence": round(overall_confidence, 3),
        "processing_notes": [
            "Processed using Qwen2-VL",
            "Image-text consistency check enabled",
            f"Category: {category}",
            f"Total images: {len(images)}"
        ]
    }


def answer_single_question(
    image_path: str,
    question: str,
    ocr_text: str = "",
    context: str = ""
) -> Dict[str, Any]:

    full_question = create_vqa_prompt(
        question=question,
        ocr_text=ocr_text,
        context=context
    )

    try:
        raw_answer = run_vqa(image_path, full_question)
        parsed = parse_vqa_response(raw_answer, question)
    except Exception as e:
        return {
            "question": question,
            "answer": "",
            "confidence": 0.0,
            "reasoning": f"VQA error: {str(e)}"
        }

    return {
        "question": question,
        "answer": parsed["answer"],
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"]
    }


def answer_three_questions_batch(
    image_paths: List[str],
    ocr_texts: List[str],
    description: str,
    questions: List[str]
) -> List[Dict[str, Any]]:
    """
    Runs Qwen2-VL on multiple images for the 3 predefined questions.
    Uses vqa_3questions.json questions by default.

    Returns a list of dictionaries, each corresponding to one image.
    """
    batch_results = []

    for idx, image_path in enumerate(image_paths):
        ocr_text = ocr_texts[idx] if idx < len(ocr_texts) else ""
        results = []

        for q in questions:
            # Add OCR comparison context for contradiction/agreement questions
            context = description
            if "تناقض" in q:
                context += f"\n\nقارن النص المرفق بالصورة (OCR):\n{ocr_text}\nأجب بنعم/لا أو صياغة قصيرة تبين وجود تناقض."
            elif "يتفق" in q:
                context += f"\n\nقارن النص المرفق بالصورة (OCR):\n{ocr_text}\nأجب بنعم إذا النص يتفق مع الصورة، لا إذا لا يتفق."

            prompt = create_vqa_prompt(
                question=q,
                ocr_text=ocr_text,
                context=context
            )

            try:
                answer = run_vqa(image_path, prompt)
                parsed = parse_vqa_response(answer, q)
            except Exception as e:
                parsed = {"answer": "", "confidence": 0.0, "reasoning": f"VQA error: {str(e)}"}

            # Boost confidence if answer is clear yes/no
            if ("نعم" in parsed["answer"] or "لا" in parsed["answer"] or "لا يوجد تناقض" in parsed["answer"]):
                parsed["confidence"] = 0.95

            results.append({
                "question": q,
                "answer": parsed["answer"],
                "confidence": parsed["confidence"],
                "reasoning": parsed["reasoning"]
            })

        batch_results.append({
            "image_path": image_path,
            "results": results
        })

    return batch_results


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def select_category_questions(
    category: str,
    all_questions: Dict[str, List[str]],
    num_questions: int = 5
) -> List[str]:
    if category not in all_questions:
        raise ValueError(f"Unknown category: {category}")

    questions = all_questions[category]
    return questions[:num_questions]


def load_image(image_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    return Image.open(image_path).convert("RGB")


def encode_image_for_api(image_path: str) -> str:
    """
    Encode image as base64 for API transmission
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(image_path, "rb") as img_file:
        encoded_bytes = base64.b64encode(img_file.read())

    return encoded_bytes.decode("utf-8")


def create_vqa_prompt(
    question: str,
    ocr_text: str = "",
    context: str = "",
    category: str = ""
) -> str:
    """
    Create prompt for VQA model
    """

    prompt_parts = []

    if category:
        prompt_parts.append(
            f"التصنيف: {category}\n"
            "أجب بدقة وبناءً فقط على ما يظهر في الصورة."
        )

    if context:
        prompt_parts.append(f"السياق:\n{context}")

    if ocr_text:
        prompt_parts.append(
            "نص مستخرج من الصورة (قد يساعدك في الإجابة):\n"
            f"{ocr_text}"
        )

    prompt_parts.append(f"السؤال:\n{question}")

    prompt_parts.append(
        "التعليمات:\n"
        "- إذا لم تكن الإجابة واضحة من الصورة، قل (غير واضح)\n"
        "- لا تفترض معلومات غير موجودة\n"
        "- أجب بإيجاز ووضوح"
    )

    return "\n\n".join(prompt_parts)


def parse_vqa_response(response: str, question: str) -> Dict[str, Any]:
    """
    Parse VQA model response into structured output
    """

    response = response.strip()

    if not response:
        return {
            "answer": "",
            "confidence": 0.0,
            "reasoning": "Empty response from model"
        }

    # Heuristic confidence estimation
    low_confidence_phrases = [
        "غير واضح",
        "لا يمكن",
        "غير معروف",
        "لا يظهر",
        "لا أستطيع"
    ]

    confidence = 0.9
    for phrase in low_confidence_phrases:
        if phrase in response:
            confidence = 0.4
            break

    # Short direct answers are usually more confident
    if len(response.split()) <= 3:
        confidence = min(confidence + 0.05, 0.95)

    return {
        "answer": response,
        "confidence": round(confidence, 2),
        "reasoning": "Parsed from Qwen2-VL response"
    }


# ============================================================================
# ERROR HANDLING
# ============================================================================

class VQAError(Exception):
    """Base exception for VQA processing errors"""
    pass


class ImageProcessingError(VQAError):
    """Exception for image processing errors"""
    pass


class APIError(VQAError):
    """Exception for API-related errors"""
    pass


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Load the 3 predefined questions from vqa_3questions.json
    QUESTIONS3_JSON = "data/vqa_3questions.json"
    
    with open(QUESTIONS3_JSON, "r", encoding="utf-8") as f:
        questions3 = json.load(f)
    
    # Example test configuration
    TEST_IMAGE_PATHS = ["ta7lel.jpg"]
    TEST_OCR_TEXTS = ["Male 20 year kidney functions , lipids profile"]
    text_description = "تحليل وظائف كبد"
    
    print(f"Loaded {len(questions3)} questions from {QUESTIONS3_JSON}")
    print(f"Questions: {questions3}")
    
    # Run VQA on images with the 3 questions
    three_q_output = answer_three_questions_batch(
        image_paths=TEST_IMAGE_PATHS,
        ocr_texts=TEST_OCR_TEXTS,
        description=text_description,
        questions=questions3
    )
    
    # Save results to JSON
    with open("vqa_three_questions.json", "w", encoding="utf-8") as f:
        json.dump(three_q_output, f, ensure_ascii=False, indent=2)
    
    print("✅ VQA processing completed successfully")
    print("📄 Results saved to vqa_three_questions.json")
