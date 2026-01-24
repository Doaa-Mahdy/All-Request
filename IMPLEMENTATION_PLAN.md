# Implementation Plan for Nafaa Charity Request Processing System

## Overview
This document outlines the step-by-step implementation plan based on the system design document, with JSON-based input/output.

---

## Phase 1: Core Infrastructure Setup

### 1.1 Simplified Project Structure
```
├── main.py                      # Main orchestrator - ALL logic + helpers
├── report_generator.py          # EXISTING - Report generation
├── quality_gate_finalized.py    # EXISTING - Image quality
├── fraud_detection.py           # EXISTING - Fraud detection  
├── reverse_image.py             # EXISTING - Image correction
├── vqa.py                       # EXISTING - Visual Q&A
├── voice_to_text.py             # EXISTING - Speech-to-text
├── config/
│   └── settings.json           # System settings + VQA questions
├── data/
│   ├── medical_products.json   # Medical items pricing (EXISTING)
│   ├── sample_input.json       # Example input
│   └── sample_output.json      # Example output
```

**Total: 8 existing specialized files + 1 new file (main.py)**

### 1.2 Dependencies
```python
# requirements.txt
openai>=1.0.0              # For GPT-4 and GPT-4V
azure-cognitiveservices-vision  # For OCR
opencv-python>=4.8.0        # Image processing
pillow>=10.0.0             # Image handling
numpy>=1.24.0              # Numerical operations
whisper                    # OpenAI Whisper for speech-to-text
python-dotenv>=1.0.0       # Environment variables
pydantic>=2.0.0            # Data validation
requests>=2.31.0           # HTTP requests
typing-extensions>=4.7.0    # Type hints
```

---

## Phase 2: Component Implementation Order

### Step 1: Core Infrastructure (Week 1)
**Priority: Critical**

**Tasks:**
1. Create `utils.py` with:
   - JSON file reader with error handling
   - JSON fmain.py` with:
   - JSON file reader with error handling
   - JSON file writer with pretty formatting
   - Basic validation functions
   - Logging setup
   - Helper functions for data transformation
   - ALL orchestration logic
2. Create unified `config/settings.json` with:
   - System configuration
   - Category-specific VQA questions (all categories)
   - Thresholds and parameters

**Files to create:**
- `main.py` (includes all utilities + orchestration)
- `config/settings.json` (unified config)
- Can read `sample_input.json` and validate structure
- Can write to `sample_output.json` with proper formatting

---
Integration (Week 1-2)
**Priority: High**

**Tasks:**
1. ✅ **USE EXISTING FILE:** `voice_to_text.py` - DO NOT MODIFY
2. Call existing voice-to-text from `main.py`
3. Add wrapper in `main.py` to handle output formatting if needed

**Existing file (DO NOT CHANGE):**
- `voice_to_text.py`

**Input:** Audio file path from input JSON
**Output:** Transcribed text + confidence + language + keywords

**Implementation:** Add function in `main.py` to call `voice_to_text.py`
- Keyword extraction for category verification

---

### Step 3: Image Quality & Fraud Detection (Week 2)
**Priority: High**

**Tasks:**
1. ✅ **USE EXISTINProcessing Integration (Week 2)
**Priority: High**

**Tasks:**
1. ✅ **USE EXISTING FILES - DO NOT MODIFY:**
   - `quality_gate_finalized.py` - Image quality checks
   - `fraud_detection.py` - Fraud detection
   - `reverse_image.py` - Image orientation/reversal correction
2. Call existing functions from `main.py`
3. Map existing outputs to required JSON format in `main.py`

**Existing files (DO NOT CHANGE):**
- `quality_gate_finalized.py`
- `fraud_detection.py`
- `reverse_image.py`

**Input:** List of image paths from input JSON
**Output:** Quality scores + fraud flags per image

**Implementation:** Add image processing logic directly in `main.py`
**Priority: High**

**Note:** ✅ **OCR text is already provided in the input JSON** for each image under the `ocr_extracted_text` field.

**Tasks:**
1. No OCR processing needed - text is pre-extracted
2. Simply read and use the `ocr_extracted_text` field from input JSON
3. Validate OCR text is present and non-empty
4. Use OCR text for VQA context and need extraction

**Integration:** OCR text available directly from input, no separate processor needed

**Input:** Already included in input JSON per image
**Output:** Use existing `ocr_extracted_text` field

---

###✅ **USE EXISTING FILE:** `vqa.py` - DO NOT MODIFY
2. Create wrapper/adapter if needed for integration
3. Define category-specific question banks:
   - Medical Aid: 20 questions
   - Education: 20 questions
   - Housing: 20 questions
   - Food: 20 questions
   - Employment: 20 questions
   - Emergency: 20 questions
4. Implement dynamic question selection (5 per image based on category)
5. Map existing VQA outputs to required JSON format

**Files to create:**
- `config/vqa_questions.json` (question banks)
- `processors/vqa_adapter.py` (wrapper if needed)

**Existing file (DO NOT CHANGE):**
- `vqa.py`

**Input:** 
- Images + category + OCR text
**Output:** 
- 5 Q&A pairs per image with confidence scores
**Output:** 
- 5 Q&A pairs per image with confidence scores

**Existing code:** `vqa.py` (can be integrated)

**Example Questions by Category:**

**Medical Aid:**
1. Is this from a recognized medical facility?
2. What is the diagnosis mentioned?
3. Is there a doctor's signature or stamp visible?
4. What is the date of this report?
5. Is surgical treatment recommended?
6. Are medication names clearly visible?
7. Are lab test results shown?
8. Is the patient name visible?
9. Do lab results indicate abnormal values?
10. Is this a recent document (within 6 months)?

**Education:**
1. Is this from an accredited educational institution?
2. What is the student's enrollment status?
3. Are tuition fees clearly stated?
4. Is the academic year mentioned?
5. Is there an official school stamp?
6. Are scholarship or financial aid details visible?
7. Is this a fee invoice or receipt?
8. Does the document show outstanding balance?
9. Is the student's name visible?
10. Is the document dated within the current academic year?

---

### Step 6: Need Extraction & Item Parser (Week 4)
**Priority: Medium-High**

**Tasks:**
1. Use LLM (GPT-4) to extract structured needs from:
   - Request text
   - OCR text
   - VQA outputs
2. Identify material items (medicines, supplies, equipment)
3. Extract quantities, specifications, duration
4. Identify non-material needs (expert consultation, legal advice)
5. ✅ **USE EXISTING FILE:** `llm.py` for LLM calls
2. Add need extraction logic in `main.py` using existing `llm.py`
3. Extract structured needs from:
   - Request text
   - OCR text
   - VQA outputs
4. Identify material items (medicines, supplies, equipment)
5. Extract quantities, specifications, duration
6. Identify non-material needs (expert consultation, legal advice)
7. Assess urgency level

**Existing file:**
- `llm.py`

**Input:** 
- Request text + OCR outputs + VQA results
**Output:** 
- List of items with quantities
- List of expert knowledge needs
- Urgency level

**Implementation:** Add LLM prompts and parsing logic in `main.py`vqa_summary}

Output JSON format:
{
  "items": [{"name": str, "quantity": int/str, "duration": str, "specs": str}],
  "info_needs": [{"topic": str, "reason": str}],
  "urgency": "low|medium|high"
}
```

---

### Step 7: Pricing Service (Week 4-5)
**Priority: Medium-High**

**Tasks:**
1. Load medical prices JSON
2. Match extracted items to price database
3. For Medical Aid: prioritize internal JSON
4. Add pricing logic directly in `main.py`
2. Load `data/medical_prices.json` (already created)
3. Match extracted items to price database
4. For Medical Aid: prioritize internal JSON
5. For other categories: implement web search fallback using `llm.py`
6. Calculate total cost ranges
7. Handle missing items gracefully
8. Generate pricing confidence scores

**Data file:**
- `data/medical_prices.json` (already created)

**Input:** 
- List of items + category
**Output:** 
- Price breakdown per item
- Total cost estimate (min, max, likely)
- Confidence scores
- Sources used

**Implementation:** Pricing functions in `main.py`, use `llm.py` for web search if needed
### Step 8: Decision Reasoning Engine (Week 5-6)
**Priority: Critical**

**Tasks:**
1. Aggregate all previous outputs
2. Use LLM to generate:
   - Accept/Reject/Need More Info decision
   - Confidence score
   - Key reasoning factors with weights
   - Risk flags
3. Idd decision logic in `main.py`
2. Use existing `llm.py` to generate:
   - Accept/Reject/Need More Info decision
   - Confidence score
   - Key reasoning factors with weights
   - Risk flags
3. Implement fairness check (compare to similar past cases)
4. Generate explanation for decision
5. Identify any inconsistencies

**Input:** 
- All previous step outputs
**Output:** 
- Decision recommendation
- Confidence score
- Reasoning breakdown
- Risk flags
- Fairness metrics

**Implementation:** Decision logic and LLM prompts in `main.py`, using `llm.py` for API calls
**Tasks:**
1. Aggregate all component outputs
2. Generate structured JSON report following sample_output.json
3. Create executive summary (1 paragraph)
4. Structure evidence analysis by image
5. Generate validity assessment
6. Create recommended actions list
7. Add metadata (models used, timestamps, etc.)

**Files to create:**
- `report_generator.py` (already exists, needs update)

**Input:** 
- All processor outputs
**Output:** 
- Complete JSON report (sample_output.json format)

**Existing code:** `report_generator.py` (can be extended)

---

### Step 10: Main Orchestrator (Week 7)
**Priority: Critical**

**Tasks:**
1. Read input JSON
2. Validate input
3. Call processors in sequence:
   ✅ **USE EXISTING FILE:** `report_generator.py` - Extend as needed
2. Aggregate all component outputs
3. Generate structured JSON report following sample_output.json
4. Create executive summary (1 paragraph)
5. Structure evidence analysis by image
6. Generate validity assessment
7. Create recommended actions list
8. Add metadata (models used, timestamps, etc.)

**Existing file:**
- `report_generator.py` (update/extend as needed)

**Input:** 
- All processor outputs from `main.py`
**Output:** 
- Complete JSON report (sample_output.json format)

**Implementation:** Call `report_generator.py` from `main.py` with all collected data
def process_request(input_json_path, output_json_path):
    # Load input
    request_data = load_json(input_json_path)
    
    # Step 1: Speech-to-text
    if request_data.is_voice:
   Create `main.py` with ALL processing logic
2. Read input JSON using `utils.py`
3. Validate input
4. Call existing files in sequence:
   - Speech-to-text (call `voice_to_text.py` if voice)
   - Image quality & fraud (call `quality_gate_finalized.py`, `fraud_detection.py`, `reverse_image.py`)
   - VQA (call `vqa.py` with category-specific questions)
   - Need extraction (use `llm.py`)
   - Pricing lookup (load JSON + use `llm.py` for web search)
   - Decision reasoning (use `llm.py`)
   - Report generation (call `report_generator.py`)
4. Handle errors gracefully
5. Write output JSON using `utils.py`
6. Log all steps

**Files to create:**
- `main.py` (single orchestrator with all logic)

**Simplified Structure:**
```python
# main.py
import utils
from voice_to_text import transcribe
from quality_gate_finalized import check_quality
from fraud_detection import detect_fraud
from reverse_image import correct_image
from vqa import answer_questions
from llm import call_llm
from report_generator import generate_report

def process_request(input_json_path, output_json_path):
    # Load input
    data = utils.load_json(input_json_path)
    
    # Step 1: Speech-to-text (if voice)
    if data['request_description']['type'] == 'voice':
        transcript = transcribe(data['request_description']['audio_file_path'])
    
    # Step 2: Image processing (quality, fraud, correction)
    for image in data['evidence_images']:
        quality = check_quality(image['image_path'])
        fraud = detect_fraud(image['image_path'])
        correct_image(image['image_path'])
    
    # Step 3: VQA (use existing vqa.py)
    vqa_results = answer_questions(images, category, questions)
    
    # Step 4: Need extraction (use llm.py)
    needs = extract_needs_with_llm(transcript, ocr_texts, vqa_results)
    
    # Step 5: Pricing (load JSON + llm for web search)
    pricing = get_pricing(needs, category)
    
    # Step 6: Decision (use llm.py)
    decision = make_decision_with_llm(all_data)
    
    # Step 7: Generate report (use report_generator.py)
    report = generate_report(all_results)
    
    # Save output
    utils.
### 4.1 Human-in-the-Loop Features
- Add staff notes section in output
- Implement override mechanism
- Track feedback for model improvement

### 4.2 Category Rules Engine
- Load category-specific rules from JSON
- Apply thresholds dynamically
- Support rule customization per category

### 4.3 Batch Processing
- Process multiple requests in sequence
- Generate summary statistics

---

## API Keys & Configuration

### Required API Keys:
1. **OpenAI API** (for GPT-4, GPT-4V, Whisper)
   - `OPENAI_API_KEY`
2. **Azure Computer Vision** (for OCR)
   - `AZURE_CV_ENDPOINT`
   - `AZURE_CV_KEY`

### Configuration File (config/settings.json):
```json
{
  "openai": {
    "api_key": "env:OPENAI_API_KEY",
    "model": "gpt-4-turbo",
    "vision_model": "gpt-4-vision-preview",
    "speech_model": "whisper-1"
  },
  "azure": {
    "cv_endpoint": "env:AZURE_CV_ENDPOINT",
    "cv_key": "env:AZURE_CV_KEY"
  },
  "thresholds": {
    "quality_min": 0.6,
    "fraud_max": "medium",
    "confidence_min": 0.7,
    "transcription_confidence_min": 0.75
  },
  "categories": [
    "Medical Aid",
    "Education",
    "Housing",
    "Food",
    "Employment",
    "Emergency"
  ]
}
```

---

## Success Criteria

### Functional:
- ✅ Can read input JSON and validate
- ✅ Can process voice/text requests
- ✅ Can analyze images using existing files
- ✅ Can answer category-specific VQA questions
- ✅ Can extract structured needs from requests
- ✅ Can lookup prices from medical_products.json (+ web search fallback)
- ✅ Can generate accept/reject decision with reasoning
- ✅ Can output structured JSON report

### Non-Functional:
- ✅ Processing time < 5 minutes per request
- ✅ 95%+ accuracy on decision recommendation
- ✅ Explainable outputs (staff can understand reasoning)
- ✅ Handles errors gracefully
- ✅ Extensible for new categories
- ✅ Minimal codebase (1 main.py + 8 existing files)

---

## Next Steps

1. **Review this plan** with the team
2. **Set up development environment** (Python 3.10+, dependencies)
3. **Create `.env` file** with API keys
4. **Start with Phase 1** (infrastructure)
5. **Implement components** in order (Phases 2-3)
6. **Test with real data** (after Week 7)
7. **Iterate based on feedback** (Weeks 8-10)

---

## Questions to Address Before Implementation

1. **API Access**: Do we have OpenAI and Azure accounts set up?
2. **Data**: Do we have real sample images and audio files for testing?
3. **Medical Database**: Is the medical prices JSON complete and up-to-date?
4. **Category Rules**: Do we have predefined rules for each category?
5. **Past Cases**: Do we have historical data for fairness comparisons?
6. **Deployment**: Where will this system run? (local, cloud, server)
7. **UI Requirements**: Will there be a web interface, or just JSON I/O?

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| API rate limits | High | Implement retry logic, batch processing |
| Poor image quality | Medium | Add preprocessing (enhance, denoise) |
| Inconsistent OCR | Medium | Use multiple OCR engines, cross-validate |
| LLM hallucinations | High | Add validation layers, confidence thresholds |
| Pricing data outdated | Medium | Add last-updated checks, web fallback |
| Long processing time | Medium | Parallelize independent steps, optimize models |
| Arabic text issues | High | Test thoroughly, use Arabic-optimized models |

---

**Let's start building! 🚀**
