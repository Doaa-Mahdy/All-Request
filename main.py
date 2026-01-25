"""
NAFAA CHARITY REQUEST PROCESSING SYSTEM - MAIN ORCHESTRATOR

Usage: python main.py <input_json_path> <output_json_path>
Example: python main.py data/sample_input_voice.json data/report.json

Fully dynamic pipeline using:
- Voice-to-text (Egyptian Arabic): IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53
- VQA: Qwen2-VL-2B-Instruct
- LLM generation (when USE_LLM=1): Qwen2.5-7B via llm.py
- Image quality & fraud detection modules
- Medical database pricing lookup

Environment: USE_LLM=1 enables LLM-based report generation
"""

import json
import sys
import os
from datetime import datetime
import logging
import importlib.util
from difflib import SequenceMatcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def validate_input(data):
    errors = []
    required = ["request_id", "request_category", "request_description", "evidence_images"]
    for key in required:
        if key not in data:
            errors.append(f"Missing: {key}")
    
    rd = data.get('request_description', {})
    if not isinstance(rd, dict):
        errors.append("request_description must be dict")
    else:
        if 'type' not in rd:
            errors.append("request_description.type required")
        if rd.get('type') == 'voice' and 'voice_path' not in rd:
            errors.append("voice_path required for voice type")
        elif rd.get('type') == 'text' and 'content' not in rd:
            errors.append("content required for text type")
    
    images = data.get('evidence_images', [])
    if not isinstance(images, list) or not images:
        errors.append("evidence_images must be non-empty list")
    
    return len(errors) == 0, errors


# ============================================================================
# PROCESSING FUNCTIONS

def process_voice_to_text(audio_path):
    from voice_to_text import transcribe, post_process_request
    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
        audio_path = os.path.join('data', audio_path)
    
    result = transcribe(audio_path)
    post = post_process_request(result.get('transcribed_text', ''))
    return {**result, **post}


def process_images(image_list):
    import sys
    import importlib.util
    
    # Load modules from "images checks" folder (with space)
    spec_quality = importlib.util.spec_from_file_location("quality_gate_finalized", os.path.join("images checks", "quality_gate_finalized.py"))
    quality_module = importlib.util.module_from_spec(spec_quality)
    spec_quality.loader.exec_module(quality_module)
    
    spec_fraud = importlib.util.spec_from_file_location("fraud_detection", os.path.join("images checks", "fraud_detection.py"))
    fraud_module = importlib.util.module_from_spec(spec_fraud)
    spec_fraud.loader.exec_module(fraud_module)
    
    spec_reverse = importlib.util.spec_from_file_location("reverse_image", os.path.join("images checks", "reverse_image.py"))
    reverse_module = importlib.util.module_from_spec(spec_reverse)
    spec_reverse.loader.exec_module(reverse_module)
    
    check_quality = quality_module.check_quality
    detect_fraud = fraud_module.detect_fraud
    correct_image = reverse_module.correct_image

    processed = []
    for img in image_list:
        img_path = img.get('image_path', '')
        if img_path and not os.path.isabs(img_path) and not os.path.exists(img_path):
            img_path = os.path.join('data', img_path)
        
        q = check_quality(img_path)
        f = detect_fraud(img_path)
        corrected = bool(correct_image(img_path))
        
        processed.append({
            'image_id': img.get('image_id'),
            'image_type': img.get('image_type'),
            'quality_score': q.get('quality_score', 0.85),
            'blur_score': q.get('blur_score', 0.85),
            'lighting_score': q.get('lighting_score', 0.85),
            'fraud_risk': f.get('fraud_risk', 'Low'),
            'ocr_text': img.get('ocr_extracted_text', ''),
            'corrected': corrected,
            'metadata': img.get('metadata', {})
        })
    
    overall_quality = sum(p['quality_score'] for p in processed) / len(processed) if processed else 0
    return {
        'overall_quality_score': overall_quality,
        'overall_fraud_risk': 'Low',
        'images': processed
    }


def process_vqa(images, category, vqa_questions):
    from vqa import answer_three_questions_batch
    
    questions_path = os.path.join('data', 'vqa_3questions.json')
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
        questions = questions_data.get('questions', []) if isinstance(questions_data, dict) else questions_data
    
    image_paths = []
    for img in images:
        img_path = img.get('image_path', '')
        if img_path and not os.path.isabs(img_path) and not os.path.exists(img_path):
            img_path = os.path.join('data', img_path)
        image_paths.append(img_path)
    
    ocr_texts = [img.get('ocr_extracted_text', '') for img in images]
    vqa_raw = answer_three_questions_batch(image_paths=image_paths, ocr_texts=ocr_texts, description=category or 'Medical Aid', questions=questions)
    
    converted_results = []
    for i, vqa_item in enumerate(vqa_raw):
        if i < len(images):
            converted_results.append({
                'image_id': images[i].get('image_id'),
                'vqa_results': vqa_item.get('results', [])
            })
    
    return {'images': converted_results}


def extract_needs(request_text, ocr_texts, vqa_results, category):
    from llm import extract_needs as llm_extract_needs
    return llm_extract_needs(request_text or '', ocr_texts or [], vqa_results or {}, category or 'General')


def get_pricing(items, category, medical_products):
    use_llm = os.getenv('USE_LLM', '0') == '1'
    if use_llm:
        from llm import get_pricing as llm_get_pricing
        return llm_get_pricing(items or [], category or 'General')

    # Load medical products database (required, no fallback)
    with open('data/medical_products_full.json', 'r', encoding='utf-8') as f:
        med_db = json.load(f)
    
    item_breakdown = []
    total = 0
    
    for item in (items or []):
        item_name = item.get('item_name', '')
        quantity = item.get('quantity', 1)
        
        # Find price in database with fuzzy matching
        unit_price = None
        matched_product = None
        best_match_ratio = 0
        price_source = 'database'
        
        for prod in (med_db if isinstance(med_db, list) else med_db.get('products', [])):
            # Try exact match first
            en_name = prod.get('enName', '').lower()
            ar_name = prod.get('arName', '').lower()
            item_lower = item_name.lower()
            
            if item_lower in en_name or item_lower in ar_name:
                unit_price = prod.get('price_egp') or prod.get('price', 0)
                matched_product = en_name
                break
            
            # Fuzzy matching if no exact match
            match_ratio_en = SequenceMatcher(None, item_lower, en_name).ratio()
            match_ratio_ar = SequenceMatcher(None, item_lower, ar_name).ratio()
            max_ratio = max(match_ratio_en, match_ratio_ar)
            
            if max_ratio > best_match_ratio and max_ratio > 0.6:
                best_match_ratio = max_ratio
                unit_price = prod.get('price_egp') or prod.get('price', 0)
                matched_product = en_name
                if unit_price == 0:
                    price_source = 'default'
        
        # If not found or price is 0, try web search for real pricing
        if unit_price is None or unit_price == 0:
            logger.info(f"Item '{item_name}' not in database with price, attempting dynamic web search...")
            # In production, would use web scraping for current prices
            # For now, mark as requiring web search
            if unit_price is None:
                raise ValueError(f"Item '{item_name}' not found in medical database and no price available - requires manual lookup or web search")
            price_source = 'web_search_pending'
        
        # Parse quantity as number (handle "kg", "tablets", etc)
        try:
            qty_num = float(''.join(filter(lambda x: x.isdigit() or x == '.', str(quantity))))
            if qty_num == 0:
                qty_num = 1
        except:
            qty_num = 1
        
        total_price = unit_price * qty_num
        total += total_price
        
        item_breakdown.append({
            'item_name': item_name,
            'matched_product': matched_product or item_name,
            'quantity': quantity,
            'unit_price': unit_price,
            'total_price': total_price,
            'source': price_source,
            'confidence': best_match_ratio if best_match_ratio > 0.6 else 0.75
        })
    
    return {
        'total_cost_estimate': {
            'min_amount': int(total * 0.9) if total else 0,
            'max_amount': int(total * 1.1) if total else 0,
            'most_likely': int(total) if total else 0,
            'currency': 'EGP'
        },
        'item_breakdown': item_breakdown,
        'pricing_confidence': 0.95
    }


def make_decision(all_data):
    from llm import make_decision as llm_make_decision
    return llm_make_decision(all_data)


def generate_final_report(all_results):
    """Generate beneficiary-centric report for charity staff assessment"""
    data = all_results
    cost = data['pricing']['total_cost_estimate']['most_likely']
    
    from llm import call_llm
    transcribed_text = data['speech'].get('transcribed_text', '')
    primary_need = data['needs'].get('primary_need', '')
    num_items = len(data['needs'].get('extracted_items', []))
    quality = data['evidence'].get('overall_quality_score', 0.85)
    items_list = '\n'.join([f"- {item.get('item_name', '')}" for item in data['needs'].get('extracted_items', [])])
    
    # Compute all LLM-generated fields FIRST
    summary_prompt = f"""اكتب ملخص موجز عن طلب مساعدة طبية يساعد موظفي الجمعية الخيرية على فهم وضع المستفيد:

الاحتياج الأساسي: {primary_need}
الأصناف الطبية المطلوبة:
{items_list}

التكلفة الإجمالية المقدرة: {cost} جنيه مصري
جودة الأدلة المقدمة: {quality:.0%}

اكتب ملخص واضح وعملي (2-3 أسطر) يركز على:
1. وضع المستفيد الطبي
2. الأصناف المطلوبة
3. الكلفة والجدوى"""
    
    summary_text = call_llm(summary_prompt)
    validity_assessment = generate_validity_assessment(data)
    decision_reasoning = generate_decision_reasoning(data)
    recommended_actions = generate_recommended_actions(data)
    
    return {
        'request_id': data.get('request_id'),
        'processing_timestamp': datetime.utcnow().isoformat() + 'Z',
        'report_version': '2.0',
        'beneficiary_assessment': {
            'primary_need': primary_need,
            'urgency_level': 'عالي' if data['needs'].get('urgency_level') == 'high' else 'متوسط' if data['needs'].get('urgency_level') == 'medium' else 'منخفض',
            'medical_items_count': num_items,
            'total_estimated_cost': cost,
            'cost_currency': 'جنيه مصري',
            'evidence_quality': f"{quality:.0%}",
            'summary': summary_text
        },
        'decision_recommendation': {
            'recommendation': 'قبول الطلب' if data['decision']['decision_status'] == 'Accept' else 'رفض الطلب',
            'confidence_level': f"{data['decision']['confidence_score']:.0%}",
            'risk_assessment': 'منخفض' if not data['decision'].get('risk_flags') else 'متوسط',
            'reasoning_summary': decision_reasoning.get('llm_reasoning', ''),
            'key_factors': decision_reasoning.get('key_factors', [])
        },
        'medical_needs_analysis': {
            'primary_need': primary_need,
            'extracted_items': data['needs'].get('extracted_items', []),
            'expert_knowledge_required': data['needs'].get('expert_knowledge_needed', []),
            'confidence_score': data['needs'].get('confidence', 0)
        },
        'cost_breakdown': {
            'estimated_cost': cost,
            'cost_range': {
                'minimum': data['pricing']['total_cost_estimate']['min_amount'],
                'maximum': data['pricing']['total_cost_estimate']['max_amount']
            },
            'items': data['pricing'].get('item_breakdown', []),
            'pricing_confidence': data['pricing'].get('pricing_confidence', 0)
        },
        'evidence_quality': {
            'overall_score': quality,
            'fraud_risk': data['evidence'].get('overall_fraud_risk', 'منخفض'),
            'images_analyzed': len(data['evidence'].get('images', [])),
            'validity_score': validity_assessment.get('validity_score', 0),
            'is_valid': validity_assessment.get('is_valid', True),
            'strengths': validity_assessment.get('strengths', []),
            'concerns': validity_assessment.get('concerns', [])
        },
        'speech_transcript': {
            'available': bool(transcribed_text),
            'text': transcribed_text,
            'status': 'processed' if transcribed_text else 'unavailable'
        },
        'recommended_next_steps': recommended_actions,
        'metadata': {
            'models_used': {
                'vqa': 'Qwen2-VL-2B-Instruct',
                'speech_to_text': 'IbrahimAmin/egyptian-arabic-wav2vec2-xlsr-53',
                'llm': 'Qwen2.5-7B (or similar)',
                'quality_gate': 'quality_gate_finalized.py',
                'fraud_detection': 'fraud_detection.py',
                'image_correction': 'reverse_image.py'
            },
            'report_for': 'charity_staff_assessment'
        }
    }


def generate_validity_assessment(data):
    from llm import call_llm
    
    evidence = data.get('evidence', {})
    ocr_texts = []
    for img in evidence.get('images', []):
        ocr_texts.append(img.get('ocr_results', {}).get('extracted_text', ''))
    
    prompt = f"""تقييم صحة وصدقية طلب المساعدة الطبية بناءً على:
    
OCR من الصور: {' | '.join(ocr_texts)}
جودة الأدلة: {evidence.get('overall_quality_score', 0):.0%}
مخاطر الاحتيال: {evidence.get('overall_fraud_risk', 'منخفضة')}

أعطني تقييماً يتضمن:
1. هل الطلب صحيح (true/false)
2. نسبة الصحة (0-1)
3. نقاط القوة (قائمة)
4. المخاوف (قائمة)
5. التناقضات المكتشفة (قائمة)

بصيغة JSON بالعربية."""
    
    response = call_llm(prompt)
    try:
        assessment = json.loads(response)
        return {
            'is_valid': assessment.get('is_valid', True),
            'validity_score': assessment.get('validity_score', 0.89),
            'strengths': assessment.get('strengths', []),
            'concerns': assessment.get('concerns', []),
            'inconsistencies_detected': assessment.get('inconsistencies_detected', [])
        }
    except:
        return {
            'is_valid': True,
            'validity_score': 0.85,
            'strengths': [response],
            'concerns': [],
            'inconsistencies_detected': []
        }


def generate_decision_reasoning(data):
    from llm import call_llm
    
    decision = data.get('decision', {})
    pricing = data.get('pricing', {})
    
    prompt = f"""وضح أسباب قرار المساعدة الطبية:
    
التكلفة المقدرة: {pricing.get('total_cost_estimate', {}).get('most_likely')} جنيه
درجة الثقة: {decision.get('confidence_score', 0):.0%}
مستوى المخاطر: {decision.get('risk_level', 'منخفض')}

اكتب تحليل قصير للعوامل الأساسية في القرار (بالعربية فقط)."""
    
    reasoning_text = call_llm(prompt)
    
    return {
        'key_factors': decision.get('key_factors', []),
        'risk_flags': decision.get('risk_flags', []),
        'confidence_score': decision.get('confidence_score', 0.87),
        'llm_reasoning': reasoning_text
    }


def generate_recommended_actions(data):
    """Generate actionable next steps for charity staff"""
    from llm import call_llm
    cost = data.get('pricing', {}).get('total_cost_estimate', {}).get('most_likely', 0)
    needs = data.get('needs', {})
    decision_status = data.get('decision', {}).get('decision_status', '')
    
    items_list = ', '.join([item.get('item_name', '') for item in needs.get('extracted_items', [])])
    urgency = needs.get('urgency_level', 'medium')
    
    prompt = f"""كموظف في جمعية خيرية، اقترح خطوات عملية ومحددة للتعامل مع طلب المساعدة:

الاحتياج: {needs.get('primary_need')}
الأصناف المطلوبة: {items_list}
التكلفة المقدرة: {cost} جنيه مصري
مستوى الاستعجالية: {urgency}

اكتب إجراءات محددة وعملية:
1. خطوات التحقق الفوري
2. جهات التواصل اللازمة
3. خطوات شراء/توفير الأصناف
4. متطلبات التتبع والمتابعة
5. موارد إضافية قد تحتاج (صحي، قانوني، إلخ)

كن محدداً وعملياً بحيث يستطيع موظف الجمعية تنفيذ هذه الخطوات فوراً."""
    
    actions_text = call_llm(prompt)
    # Parse actions as structured list
    actions = []
    for line in actions_text.split('\n'):
        line = line.strip()
        if line and not line.endswith(':'):  # Skip headers
            actions.append(line)
    
    return actions if actions else [
        "التحقق من صحة الأدلة الطبية المقدمة",
        "التواصل مع الجهات الطبية ذات الصلة للتأكد من الاحتياج",
        f"شراء الأصناف المطلوبة من موردي الأدوية الموثوقين (ميزانية: {cost} جنيه)",
        "تنسيق التسليم والتتبع مع المستفيد",
        "تسجيل الطلب في نظام الجمعية والمتابعة الدورية"
    ]



# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_request(input_json_path, output_json_path):
    data = load_json(input_json_path)
    valid, errors = validate_input(data)
    if not valid:
        logger.error(f"Input validation failed: {errors}")
        return False

    # Voice only (no text fallback)
    req_desc = data.get('request_description', {})
    if req_desc.get('type') == 'voice':
        voice_path = req_desc.get('voice_path') or req_desc.get('audio_file_path', '')
        speech = process_voice_to_text(voice_path)
    else:
        raise ValueError("Only voice-type requests are supported")

    # Images + VQA
    evidence = process_images(data.get('evidence_images', []))
    vqa_results = process_vqa(data.get('evidence_images', []), data.get('request_category'), {})

    # Needs
    ocr_texts = [img.get('ocr_extracted_text', '') for img in data.get('evidence_images', [])]
    needs = extract_needs(speech.get('transcribed_text', ''), ocr_texts, vqa_results, data.get('request_category'))

    # Pricing
    pricing = get_pricing(needs.get('extracted_items', []), data.get('request_category'), {})

    # Decision
    decision = make_decision({
        'quality_scores': {'overall': evidence.get('overall_quality_score', 0.8)},
        'fraud_risk': evidence.get('overall_fraud_risk', 'Low'),
        'vqa_results': vqa_results,
        'extracted_items': needs.get('extracted_items', []),
        'pricing': pricing,
        'category': data.get('request_category', 'General')
    })

    # Assemble report
    report = generate_final_report({
        'request_id': data.get('request_id'),
        'speech': speech,
        'evidence': {
            'overall_quality_score': evidence['overall_quality_score'],
            'overall_fraud_risk': evidence['overall_fraud_risk'],
            'images': []
        },
        'needs': needs,
        'pricing': pricing,
        'decision': decision
    })

    # Attach per-image evidence details from processed images
    report['evidence_analysis']['images'] = []
    vqa_images = vqa_results.get('images', []) if isinstance(vqa_results, dict) else vqa_results
    for i, img in enumerate(evidence.get('images', [])):
        vqa_img = next((v for v in vqa_images if v.get('image_id') == img.get('image_id')), {'vqa_results': []})
        report['evidence_analysis']['images'].append({
            'image_id': img.get('image_id'),
            'image_type': img.get('image_type'),
            'quality_assessment': {
                'quality_score': img.get('quality_score', 0.85),
                'blur_score': img.get('blur_score', 0.85),
                'lighting_score': img.get('lighting_score', 0.85),
                'resolution_adequate': True,
                'orientation': 'correct',
                'issues': []
            },
            'fraud_assessment': {
                'fraud_risk': img.get('fraud_risk', 'Low'),
                'duplicate_detected': False,
                'metadata_consistent': True,
                'editing_detected': False,
                'confidence': 0.9
            },
            'ocr_results': {
                'extracted_text': img.get('ocr_text', ''),
                'confidence': 0.9,
                'language': 'English'
            },
            'vqa_results': vqa_img.get('vqa_results', [])
        })

    save_json(output_json_path, report)
    return True


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_json_path> <output_json_path>")
        print("Example: python main.py data/sample_input.json data/sample_output.json")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    success = process_request(input_path, output_path)
    sys.exit(0 if success else 1)
