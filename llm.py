"""
================================================================================
LLM INTEGRATION MODULE USING QWEN3-1.7
================================================================================

PURPOSE:
    - Integrate with Qwen3-1.7 LLM (Qwen2.5-7B)
    - Provide functions for need extraction, decision reasoning, pricing lookup
    - Handle web search using DuckDuckGo for non-medical items
    - Integrate with medical_products_full.json for medical pricing

HOW TO USE:
    from llm import call_llm, extract_needs, make_decision, get_pricing
    
    result = call_llm(prompt="Extract medical items from this text...")
    needs = extract_needs(text, ocr_texts, vqa_results, category)
    pricing = get_pricing(needs['extracted_items'], category)
    decision = make_decision(all_data)

KEY FUNCTIONS:

1. call_llm(prompt, temperature=0.7, max_tokens=2000)
   - Make generic LLM API call to Qwen3-1.7
   - Input: prompt string
   - Output: LLM response string
   
2. extract_needs(request_text, ocr_texts, vqa_results, category)
   - Extract structured needs from request evidence
   - Input: text content, extracted text, VQA results, category
   - Output: dict with items, expert needs, urgency
   
3. get_item_price(item_name, category)
   - Lookup item price in medical_products_full.json
   - Input: item name (English or Arabic), category
   - Output: price in EGP, source, confidence
   
4. search_web(query, context="Egypt")
   - Search web using DuckDuckGo for pricing info
   - Input: search query, geographic context
   - Output: search results with extracted prices
   
5. get_pricing(extracted_items, category, use_web_search=True)
   - Get complete pricing breakdown for all items
   - Input: items from extract_needs(), category
   - Output: price breakdown, total costs, confidence scores
   
6. make_decision(all_data)
   - Generate accept/reject decision with reasoning
   - Input: aggregated processing results
   - Output: decision, confidence, reasoning, risk flags

DEPENDENCIES:
    - transformers >= 4.40.0
    - torch >= 2.0.0
    - duckduckgo_search >= 3.8.0 (optional, for web search)

ENVIRONMENT VARIABLES:
    - Uses Hugging Face transformers (no API key needed)

DATA FILES:
    - data/medical_products_full.json - Medical items with prices
    - data/vqa_3questions.json - VQA questions for image analysis

EXAMPLE USAGE:

    # Extract needs from request
    needs = extract_needs(
        request_text="My child needs kidney surgery",
        ocr_texts=["Diagnosis: CKD Stage 3..."],
        vqa_results={...},
        category="Medical Aid"
    )
    # Output:
    # {
    #     'primary_need': 'Medical treatment for kidney disease',
    #     'extracted_items': [
    #         {'item_name': 'Erythropoietin', 'quantity': 72, ...}
    #     ],
    #     'urgency_level': 'High'
    # }

    # Get pricing information
    pricing = get_pricing(
        extracted_items=needs['extracted_items'],
        category="Medical Aid"
    )
    # Output:
    # {
    #     'items_pricing': [
    #         {
    #             'item_name': 'Erythropoietin',
    #             'quantity': 72,
    #             'unit_price_egp': 500.0,
    #             'total_price_egp': 36000.0,
    #             'source': 'medical_database',
    #             'confidence': 0.95
    #         }
    #     ],
    #     'total_cost_likely': 36000.0,
    #     'pricing_confidence': 1.0
    # }

    # Make decision
    decision = make_decision({
        'quality_scores': {...},
        'fraud_risk': 'Low',
        'pricing': pricing,
        'category': 'Medical Aid'
    })
    # Output:
    # {
    #     'status': 'Accept',
    #     'confidence_score': 0.87,
    #     'risk_flags': [],
    #     'reasoning': '...'
    # }

================================================================================
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Qwen3-1.7 Model
try:
    logger.info("Loading Qwen3-1.7 model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Using Qwen2.5 as closest available to Qwen3-1.7
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    logger.info("Qwen3-1.7 model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Qwen3-1.7 model: {str(e)}")
    model = None
    tokenizer = None

DEVICE = device if model else "cpu"


# ============================================================================
# MAIN LLM FUNCTIONS
# ============================================================================

def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
    """
    Make a generic LLM API call to Qwen3-1.7
    
    Args:
        prompt (str): The prompt for the LLM
        temperature (float): Creativity level (0-1), default 0.7
        max_tokens (int): Maximum response length, default 2000
        
    Returns:
        str: LLM response text
        
    Example:
        response = call_llm("Summarize this medical report in 2 sentences")
        print(response)
    """
    if model is None or tokenizer is None:
        logger.error("Qwen3-1.7 model not loaded")
        return ""
    
    try:
        logger.info(f"Calling Qwen3-1.7 with prompt ({len(prompt)} chars)")
        
        # Prepare messages for chat format
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Encode messages
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True
            )
        
        # Decode response
        generated_ids = generated_ids[:, model_inputs.input_ids.shape[-1]:]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        logger.info(f"LLM response generated ({len(response)} chars)")
        return response.strip()
        
    except Exception as e:
        logger.error(f"Error calling LLM: {str(e)}")
        return ""


def extract_needs(
    request_text: str,
    ocr_texts: List[str],
    vqa_results: Dict,
    category: str
) -> Dict[str, Any]:
    """
    Extract structured needs from request text and evidence
    
    Args:
        request_text (str): Original request text
        ocr_texts (list): List of OCR-extracted texts from images
        vqa_results (dict): VQA results from images
        category (str): Request category
        
    Returns:
        dict: {
            'primary_need': str,
            'extracted_items': [
                {
                    'item_name': str,
                    'quantity': int/str,
                    'duration': str,
                    'specifications': str,
                    'medical_necessity': str (if Medical Aid)
                }
            ],
            'expert_knowledge_needed': [
                {
                    'topic': str,
                    'reason': str,
                    'addressed': bool
                }
            ],
            'urgency_level': str  # 'low', 'medium', 'high'
        }
    """
    logger.info(f"Extracting needs for category: {category}")
    
    # Prepare context
    ocr_context = "\n".join(ocr_texts) if ocr_texts else "No OCR text available"
    vqa_context = json.dumps(vqa_results, ensure_ascii=False, indent=2) if vqa_results else "No VQA results"
    
    # Create extraction prompt
    extraction_prompt = create_extraction_prompt(request_text, ocr_texts, category)
    
    try:
        # Call LLM for extraction
        llm_response = call_llm(extraction_prompt, temperature=0.5, max_tokens=2000)
        
        # Parse JSON response
        is_valid, parsed_data = validate_llm_response(llm_response, 'json')
        
        if is_valid and isinstance(parsed_data, dict):
            result = {
                'primary_need': parsed_data.get('primary_need', 'Unable to determine primary need'),
                'extracted_items': parsed_data.get('extracted_items', []),
                'expert_knowledge_needed': parsed_data.get('expert_knowledge_needed', []),
                'urgency_level': parsed_data.get('urgency_level', 'medium'),
                'confidence': parsed_data.get('confidence', 0.7)
            }
            logger.info(f"Extracted {len(result['extracted_items'])} items")
            return result
        else:
            # Fallback: try to extract manually
            logger.warning("LLM response parsing failed, using fallback extraction")
            return fallback_extract_needs(request_text, ocr_texts, category)
            
    except Exception as e:
        logger.error(f"Error extracting needs: {str(e)}")
        return {
            'primary_need': 'Extraction failed',
            'extracted_items': [],
            'expert_knowledge_needed': [],
            'urgency_level': 'medium',
            'confidence': 0.0
        }


def make_decision(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate accept/reject decision with reasoning
    
    Args:
        all_data (dict): Aggregated data from all processing steps including:
            - quality_scores: image quality assessments
            - fraud_risk: fraud detection results
            - vqa_results: visual Q&A answers
            - extracted_items: parsed items and needs
            - pricing: cost breakdown
            - category: request category
            
    Returns:
        dict: {
            'decision_status': str,  # 'Accept', 'Reject', 'Needs More Info'
            'confidence_score': float,  # 0-1
            'key_factors': [...],
            'risk_flags': [str],
            'reasoning': str,
            'concerns': [str],
            'recommendations': [str]
        }
    """
    logger.info("Generating decision with Qwen3-1.7")
    
    try:
        # Create decision prompt
        decision_prompt = create_decision_prompt(all_data)
        
        # Call LLM for decision
        llm_response = call_llm(decision_prompt, temperature=0.3, max_tokens=2000)
        
        # Parse response
        is_valid, parsed_data = validate_llm_response(llm_response, 'json')
        
        if is_valid and isinstance(parsed_data, dict):
            result = {
                'decision_status': parsed_data.get('decision_status', 'Needs More Info'),
                'confidence_score': parsed_data.get('confidence_score', 0.5),
                'key_factors': parsed_data.get('key_factors', []),
                'risk_flags': parsed_data.get('risk_flags', []),
                'reasoning': parsed_data.get('reasoning', 'Unable to generate reasoning'),
                'concerns': parsed_data.get('concerns', []),
                'recommendations': parsed_data.get('recommendations', [])
            }
            logger.info(f"Decision generated: {result['decision_status']} (confidence: {result['confidence_score']})")
            return result
        else:
            logger.warning("Decision parsing failed, using rule-based fallback")
            return fallback_make_decision(all_data)
            
    except Exception as e:
        logger.error(f"Error making decision: {str(e)}")
        return {
            'decision_status': 'Needs More Info',
            'confidence_score': 0.0,
            'key_factors': [],
            'risk_flags': ['LLM processing error'],
            'reasoning': f'Error during decision generation: {str(e)}',
            'concerns': ['Unable to process request'],
            'recommendations': ['Review manually']
        }


def validate_llm_response(response: str, expected_format: str) -> Tuple[bool, Any]:
    """
    Validate and parse LLM response
    
    Args:
        response (str): LLM response text
        expected_format (str): Expected format ('json', 'text', 'list')
        
    Returns:
        tuple: (is_valid: bool, parsed_data: Any)
    """
    if not response:
        logger.warning("Empty LLM response")
        return False, None
    
    try:
        if expected_format == 'json':
            # Try to extract JSON from response
            # Look for JSON block between curly braces
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group()
                parsed = json.loads(json_str)
                logger.info("JSON response validated successfully")
                return True, parsed
            else:
                logger.warning("No JSON found in response")
                return False, None
        
        elif expected_format == 'list':
            # Try to extract list
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group()
                parsed = json.loads(json_str)
                logger.info("List response validated successfully")
                return True, parsed
            else:
                return False, None
        
        else:  # 'text' format
            return True, response.strip()
            
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing error: {str(e)}")
        return False, None
    except Exception as e:
        logger.error(f"Response validation error: {str(e)}")
        return False, None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_extraction_prompt(
    request_text: str,
    ocr_texts: List[str],
    category: str
) -> str:
    """
    Create LLM prompt for need extraction
    """
    ocr_context = "\n".join(ocr_texts) if ocr_texts else "No OCR text available"
    
    prompt = f"""You are an expert at analyzing charity requests. Extract structured needs from the following request.

CATEGORY: {category}

REQUEST TEXT:
{request_text}

OCR EXTRACTED TEXT FROM IMAGES:
{ocr_context}

Please analyze the request and respond with a JSON object containing:
1. primary_need: The main need expressed
2. extracted_items: List of specific items/services needed (with name, quantity, duration, specifications)
3. expert_knowledge_needed: List of expert consultations or knowledge areas needed
4. urgency_level: 'low', 'medium', or 'high'
5. confidence: Your confidence score (0-1)

Respond ONLY with valid JSON, no additional text.

Example format:
{{
  "primary_need": "Medical treatment for kidney disease",
  "extracted_items": [
    {{"item_name": "Erythropoietin 4000 IU", "quantity": 72, "duration": "6 months", "specifications": "Pre-filled syringes"}}
  ],
  "expert_knowledge_needed": [
    {{"topic": "Nephrologist consultation", "reason": "Treatment planning"}}
  ],
  "urgency_level": "high",
  "confidence": 0.85
}}"""
    
    return prompt


def create_decision_prompt(all_data: Dict) -> str:
    """
    Create LLM prompt for decision making
    """
    data_str = json.dumps(all_data, ensure_ascii=False, indent=2)
    
    prompt = f"""You are an expert reviewer for a charity organization. Based on the following comprehensive analysis of a charity request, make an accept/reject decision.

ANALYSIS DATA:
{data_str}

Please analyze all factors including:
- Document quality and validity
- Fraud risk assessment
- Request legitimacy and urgency
- Item relevance and pricing reasonableness
- Category-specific requirements

Respond with a JSON object containing:
1. decision_status: 'Accept', 'Reject', or 'Needs More Info'
2. confidence_score: Your confidence (0-1)
3. key_factors: List of important factors considered
4. risk_flags: List of concerning flags if any
5. reasoning: Detailed explanation of decision
6. concerns: Specific concerns raised
7. recommendations: Recommended next steps

Respond ONLY with valid JSON, no additional text.

Example format:
{{
  "decision_status": "Accept",
  "confidence_score": 0.87,
  "key_factors": ["Document authenticity verified", "Item prices reasonable", "Urgency justified"],
  "risk_flags": [],
  "reasoning": "Clear legitimate medical need with valid documentation. Request is reasonable and within scope.",
  "concerns": [],
  "recommendations": ["Process payment", "Arrange follow-up"]
}}"""
    
    return prompt


def fallback_extract_needs(request_text: str, ocr_texts: List[str], category: str) -> Dict[str, Any]:
    """
    Fallback extraction when LLM fails
    """
    logger.warning("Using fallback need extraction")
    
    # Simple keyword-based extraction
    urgency_keywords = ['urgent', 'emergency', 'immediately', 'critical', 'عاجل', 'طارئ']
    medical_keywords = ['medical', 'treatment', 'doctor', 'hospital', 'medicine', 'علاج', 'مستشفى']
    
    combined_text = (request_text + " " + " ".join(ocr_texts)).lower()
    
    urgency = 'high' if any(k in combined_text for k in urgency_keywords) else 'medium'
    is_medical = any(k in combined_text for k in medical_keywords)
    
    return {
        'primary_need': f'{category} support',
        'extracted_items': [],
        'expert_knowledge_needed': [],
        'urgency_level': urgency,
        'confidence': 0.3
    }


def fallback_make_decision(all_data: Dict) -> Dict[str, Any]:
    """
    Fallback decision when LLM fails - uses rule-based approach
    """
    logger.warning("Using fallback decision making")
    
    # Simple rule-based decision
    quality = all_data.get('quality_scores', {}).get('overall', 0.5)
    fraud_risk = all_data.get('fraud_risk', 'Medium')
    category = all_data.get('category', 'General')
    
    # Decision logic
    if fraud_risk == 'High':
        status = 'Reject'
        confidence = 0.8
    elif quality < 0.5:
        status = 'Needs More Info'
        confidence = 0.6
    elif quality > 0.75:
        status = 'Accept'
        confidence = 0.7
    else:
        status = 'Needs More Info'
        confidence = 0.5
    
    return {
        'decision_status': status,
        'confidence_score': confidence,
        'key_factors': [f'Quality score: {quality}', f'Fraud risk: {fraud_risk}'],
        'risk_flags': ['Rule-based fallback used'],
        'reasoning': f'Based on quality ({quality}) and fraud assessment ({fraud_risk})',
        'concerns': ['LLM unavailable, using rule-based fallback'],
        'recommendations': ['Review manually if uncertain']
    }


def get_item_price(item_name: str, category: str) -> Dict[str, Any]:
    """
    Lookup item price from medical products database
    
    Args:
        item_name (str): Name of the item (English or Arabic)
        category (str): Request category
        
    Returns:
        dict: {
            'item_name': str,
            'found': bool,
            'price_egp': float or None,
            'price_usd': float or None,
            'source': str,
            'confidence': float,
            'alternatives': [str]
        }
    """
    try:
        # Load medical products database
        medical_db_path = os.path.join(os.path.dirname(__file__), 'data', 'medical_products_full.json')
        
        if not os.path.exists(medical_db_path):
            logger.warning(f"Medical products database not found at {medical_db_path}")
            return {
                'item_name': item_name,
                'found': False,
                'price_egp': None,
                'price_usd': None,
                'source': 'database_not_found',
                'confidence': 0.0,
                'alternatives': []
            }
        
        # Load database (limit to reasonable size to avoid memory issues)
        with open(medical_db_path, 'r', encoding='utf-8') as f:
            products = json.load(f)
        
        if not isinstance(products, list):
            products = list(products.values()) if isinstance(products, dict) else []
        
        logger.info(f"Searching {len(products)} products for '{item_name}'")
        
        # Search for exact match (English)
        for product in products:
            if product.get('enName', '').lower() == item_name.lower():
                price = product.get('price', {})
                return {
                    'item_name': item_name,
                    'found': True,
                    'price_egp': price.get('egp'),
                    'price_usd': price.get('usd'),
                    'source': 'medical_database',
                    'confidence': 0.95,
                    'alternatives': [],
                    'product_id': product.get('key'),
                    'description': product.get('description', '')[:200]
                }
            
            # Search for Arabic name match
            if product.get('arName', '').lower() == item_name.lower():
                price = product.get('price', {})
                return {
                    'item_name': item_name,
                    'found': True,
                    'price_egp': price.get('egp'),
                    'price_usd': price.get('usd'),
                    'source': 'medical_database',
                    'confidence': 0.95,
                    'alternatives': [],
                    'product_id': product.get('key'),
                    'description': product.get('description', '')[:200]
                }
        
        # Search for partial match
        item_lower = item_name.lower()
        partial_matches = []
        for product in products:
            en_name = product.get('enName', '').lower()
            ar_name = product.get('arName', '').lower()
            if item_lower in en_name or item_lower in ar_name or en_name in item_lower or ar_name in item_lower:
                partial_matches.append(product)
        
        if partial_matches:
            # Return top match with alternatives
            top_match = partial_matches[0]
            price = top_match.get('price', {})
            alternatives = [p.get('enName', p.get('arName', 'Unknown')) for p in partial_matches[1:4]]
            
            logger.info(f"Found {len(partial_matches)} partial matches for '{item_name}'")
            return {
                'item_name': item_name,
                'found': True,
                'price_egp': price.get('egp'),
                'price_usd': price.get('usd'),
                'source': 'medical_database_partial_match',
                'confidence': 0.7,
                'alternatives': alternatives,
                'product_id': top_match.get('key'),
                'description': top_match.get('description', '')[:200]
            }
        
        logger.warning(f"No match found for item: {item_name}")
        return {
            'item_name': item_name,
            'found': False,
            'price_egp': None,
            'price_usd': None,
            'source': 'not_found',
            'confidence': 0.0,
            'alternatives': []
        }
        
    except Exception as e:
        logger.error(f"Error looking up price for {item_name}: {str(e)}")
        return {
            'item_name': item_name,
            'found': False,
            'price_egp': None,
            'price_usd': None,
            'source': f'error: {str(e)}',
            'confidence': 0.0,
            'alternatives': []
        }


def search_web(query: str, context: str = "Egypt") -> Dict[str, Any]:
    """
    Search web for pricing information using DuckDuckGo
    
    Args:
        query (str): Search query
        context (str): Geographic context (e.g., "Egypt")
        
    Returns:
        dict: {
            'query': str,
            'found': bool,
            'results': [
                {
                    'title': str,
                    'url': str,
                    'snippet': str,
                    'price_mentioned': float or None,
                    'relevance': float
                }
            ],
            'estimated_price_egp': float or None,
            'estimated_price_usd': float or None,
            'confidence': float,
            'source': str
        }
    """
    try:
        # Try to import duckduckgo_search
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            logger.warning("duckduckgo_search not installed, returning empty results")
            return {
                'query': query,
                'found': False,
                'results': [],
                'estimated_price_egp': None,
                'estimated_price_usd': None,
                'confidence': 0.0,
                'source': 'duckduckgo_not_installed'
            }
        
        # Enhance query with context
        search_query = f"{query} price {context}"
        logger.info(f"Searching web for: {search_query}")
        
        # Perform search
        with DDGS() as ddgs:
            results = list(ddgs.text(search_query, max_results=5))
        
        if not results:
            logger.warning(f"No web search results for: {search_query}")
            return {
                'query': query,
                'found': False,
                'results': [],
                'estimated_price_egp': None,
                'estimated_price_usd': None,
                'confidence': 0.0,
                'source': 'web_search_no_results'
            }
        
        # Process results
        processed_results = []
        prices_found = []
        
        for result in results:
            processed = {
                'title': result.get('title', ''),
                'url': result.get('href', ''),
                'snippet': result.get('body', '')[:300],
                'price_mentioned': None,
                'relevance': 0.8
            }
            
            # Try to extract price from snippet
            import re
            price_patterns = [
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:EGP|ج\.م|جنيه)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:USD|$|dollar)',
                r'₩\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:LE|pound)'
            ]
            
            snippet_lower = (result.get('body', '') + result.get('title', '')).lower()
            for pattern in price_patterns:
                match = re.search(pattern, snippet_lower, re.IGNORECASE)
                if match:
                    try:
                        price_str = match.group(1).replace(',', '')
                        processed['price_mentioned'] = float(price_str)
                        prices_found.append(float(price_str))
                        processed['relevance'] = 0.9
                        break
                    except (ValueError, IndexError):
                        pass
            
            processed_results.append(processed)
        
        # Calculate estimated price
        estimated_price = None
        if prices_found:
            estimated_price = sum(prices_found) / len(prices_found)
        
        logger.info(f"Web search found {len(results)} results, {len(prices_found)} with prices")
        return {
            'query': query,
            'found': len(results) > 0,
            'results': processed_results,
            'estimated_price_egp': estimated_price if estimated_price and estimated_price < 100000 else None,
            'estimated_price_usd': (estimated_price / 30) if estimated_price and estimated_price < 100000 else None,
            'confidence': 0.6 if prices_found else 0.4,
            'source': 'web_search_duckduckgo',
            'num_results': len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching web for {query}: {str(e)}")
        return {
            'query': query,
            'found': False,
            'results': [],
            'estimated_price_egp': None,
            'estimated_price_usd': None,
            'confidence': 0.0,
            'source': f'error: {str(e)}'
        }


def get_pricing(
    extracted_items: List[Dict],
    category: str,
    use_web_search: bool = True
) -> Dict[str, Any]:
    """
    Get pricing information for extracted items
    
    Args:
        extracted_items (list): Items from extract_needs()
        category (str): Request category
        use_web_search (bool): Whether to use web search for non-medical items
        
    Returns:
        dict: {
            'items_pricing': [
                {
                    'item_name': str,
                    'quantity': int/str,
                    'unit_price_egp': float or None,
                    'total_price_egp': float or None,
                    'source': str,
                    'confidence': float
                }
            ],
            'total_cost_min': float,
            'total_cost_likely': float,
            'total_cost_max': float,
            'pricing_confidence': float,
            'sources_used': [str]
        }
    """
    logger.info(f"Getting pricing for {len(extracted_items)} items in category: {category}")
    
    items_pricing = []
    prices_collected = []
    sources_used = set()
    
    for item in extracted_items:
        item_name = item.get('item_name', '')
        quantity = item.get('quantity', 1)
        
        # Try to get price from medical database first
        if category == 'Medical Aid':
            price_result = get_item_price(item_name, category)
            sources_used.add(price_result.get('source', 'unknown'))
            
            if price_result['found'] and price_result.get('price_egp'):
                unit_price = price_result['price_egp']
                total_price = unit_price * (quantity if isinstance(quantity, (int, float)) else 1)
                items_pricing.append({
                    'item_name': item_name,
                    'quantity': quantity,
                    'unit_price_egp': unit_price,
                    'total_price_egp': total_price,
                    'source': price_result['source'],
                    'confidence': price_result['confidence'],
                    'product_id': price_result.get('product_id')
                })
                prices_collected.append(total_price)
                logger.info(f"Found price for {item_name}: {unit_price} EGP")
            else:
                # Try web search if not found in database
                if use_web_search:
                    logger.info(f"Price not in database, searching web for {item_name}")
                    web_result = search_web(f"{item_name} price", context="Egypt")
                    sources_used.add(web_result.get('source', 'unknown'))
                    
                    if web_result['found'] and web_result.get('estimated_price_egp'):
                        unit_price = web_result['estimated_price_egp']
                        total_price = unit_price * (quantity if isinstance(quantity, (int, float)) else 1)
                        items_pricing.append({
                            'item_name': item_name,
                            'quantity': quantity,
                            'unit_price_egp': unit_price,
                            'total_price_egp': total_price,
                            'source': web_result['source'],
                            'confidence': web_result['confidence']
                        })
                        prices_collected.append(total_price)
                        logger.info(f"Found web price for {item_name}: {unit_price} EGP")
                    else:
                        # Add item without price
                        items_pricing.append({
                            'item_name': item_name,
                            'quantity': quantity,
                            'unit_price_egp': None,
                            'total_price_egp': None,
                            'source': 'price_not_found',
                            'confidence': 0.0
                        })
                        logger.warning(f"No price found for {item_name}")
                else:
                    # Add item without price
                    items_pricing.append({
                        'item_name': item_name,
                        'quantity': quantity,
                        'unit_price_egp': None,
                        'total_price_egp': None,
                        'source': 'web_search_disabled',
                        'confidence': 0.0
                    })
        else:
            # For non-medical categories, try web search first
            logger.info(f"Non-medical category, searching web for {item_name}")
            web_result = search_web(f"{item_name} price", context="Egypt")
            sources_used.add(web_result.get('source', 'unknown'))
            
            if web_result['found'] and web_result.get('estimated_price_egp'):
                unit_price = web_result['estimated_price_egp']
                total_price = unit_price * (quantity if isinstance(quantity, (int, float)) else 1)
                items_pricing.append({
                    'item_name': item_name,
                    'quantity': quantity,
                    'unit_price_egp': unit_price,
                    'total_price_egp': total_price,
                    'source': web_result['source'],
                    'confidence': web_result['confidence']
                })
                prices_collected.append(total_price)
                logger.info(f"Found web price for {item_name}: {unit_price} EGP")
            else:
                # Add item without price
                items_pricing.append({
                    'item_name': item_name,
                    'quantity': quantity,
                    'unit_price_egp': None,
                    'total_price_egp': None,
                    'source': 'price_not_found',
                    'confidence': 0.0
                })
                logger.warning(f"No price found for {item_name}")
    
    # Calculate total costs
    if prices_collected:
        total_cost_likely = sum(prices_collected)
        total_cost_min = total_cost_likely * 0.9  # 10% lower estimate
        total_cost_max = total_cost_likely * 1.2  # 20% higher estimate
    else:
        total_cost_likely = None
        total_cost_min = None
        total_cost_max = None
    
    pricing_confidence = len(prices_collected) / len(items_pricing) if items_pricing else 0.0
    
    result = {
        'items_pricing': items_pricing,
        'total_cost_min': total_cost_min,
        'total_cost_likely': total_cost_likely,
        'total_cost_max': total_cost_max,
        'pricing_confidence': pricing_confidence,
        'sources_used': list(sources_used),
        'items_with_price': len(prices_collected),
        'items_without_price': len(items_pricing) - len(prices_collected)
    }
    
    logger.info(f"Pricing complete: {result['items_with_price']}/{len(items_pricing)} items priced")
    return result


def get_category_guidelines(category: str) -> Dict:
    """
    Get category-specific guidelines for processing
    """
    guidelines = {
        'Medical Aid': {
            'required_documents': ['Medical diagnosis', 'Doctor prescription', 'Cost estimate'],
            'max_reasonable_cost': 50000,
            'typical_items': ['Medicines', 'Surgery', 'Therapy', 'Equipment']
        },
        'Education': {
            'required_documents': ['School certificate', 'Fee invoice', 'Enrollment letter'],
            'max_reasonable_cost': 30000,
            'typical_items': ['Tuition', 'Books', 'Materials', 'Uniforms']
        },
        'Housing': {
            'required_documents': ['Property deed', 'Repair estimate', 'ID'],
            'max_reasonable_cost': 100000,
            'typical_items': ['Repairs', 'Rent deposit', 'Materials']
        },
        'Food': {
            'required_documents': ['Family ID', 'Living situation proof'],
            'max_reasonable_cost': 5000,
            'typical_items': ['Groceries', 'Food supplies']
        },
        'Emergency': {
            'required_documents': ['Evidence of emergency', 'Immediate need'],
            'max_reasonable_cost': 20000,
            'typical_items': ['Immediate relief', 'Emergency services']
        }
    }
    
    return guidelines.get(category, guidelines['Emergency'])


# ============================================================================
# ERROR HANDLING
# ============================================================================

class LLMError(Exception):
    """Base exception for LLM processing errors"""
    pass


class APIError(LLMError):
    """Exception for API-related errors"""
    pass


class ValidationError(LLMError):
    """Exception for validation errors"""
    pass


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("LLM Module - Qwen3-1.7 Integration")
    print("Usage: from llm import extract_needs, make_decision, call_llm, get_pricing")
    print("\nAvailable functions:")
    print("  1. call_llm(prompt, temperature, max_tokens)")
    print("     - Generic LLM API calls")
    print("  2. extract_needs(request_text, ocr_texts, vqa_results, category)")
    print("     - Extract structured needs from evidence")
    print("  3. get_item_price(item_name, category)")
    print("     - Lookup prices in medical_products_full.json")
    print("  4. search_web(query, context='Egypt')")
    print("     - Search web using DuckDuckGo")
    print("  5. get_pricing(extracted_items, category, use_web_search)")
    print("     - Get complete pricing breakdown")
    print("  6. make_decision(all_data)")
    print("     - Generate accept/reject decision")
    
    # Test the LLM if model is loaded
    if model and tokenizer:
        print("\n✅ Qwen3-1.7 model is ready")
        test_prompt = "What are the main types of medical aid?"
        print(f"\nTest prompt: {test_prompt}")
        result = call_llm(test_prompt, max_tokens=200)
        print(f"Response: {result[:200]}..." if len(result) > 200 else f"Response: {result}")
        
        print("\n✅ Testing pricing functions...")
        # Test pricing lookup
        price_result = get_item_price("Ibuprofen", "Medical Aid")
        print(f"Price lookup test: {price_result['source']} - Found: {price_result['found']}")
        
        # Test web search
        web_result = search_web("insulin Egypt", context="Egypt")
        print(f"Web search test: {web_result['source']} - Results: {web_result['num_results'] if 'num_results' in web_result else 'N/A'}")
    else:
        print("\n❌ Qwen3-1.7 model failed to load")

