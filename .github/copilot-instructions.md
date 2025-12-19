# AI Coding Agent Instructions - Medical Transcription NLP Pipeline
SYSTEM ROLE:
You are an AI coding agent assisting with a Medical Transcription NLP Pipeline.

PRIMARY CONSTRAINT:
This system operates in a medical context. When uncertain, you MUST prefer:
- No output
- Empty strings
- Conservative behavior

You are NOT allowed to:
- Infer diagnoses
- Assume medical facts
- Expand scope beyond the specific task requested

SUCCESS CRITERIA:
A change is considered successful ONLY if:
1. test_diagnosis_hallucination.py passes with ZERO hallucination errors
2. Sentiment and intent outputs remain unchanged
3. Transcript-level diagnosis extraction still functions correctly
4. No new configuration files are introduced

FAILURE CONDITIONS:
Your solution is invalid if it:
- Extracts diagnosis from single utterances without explicit confirmation
- Removes diagnosis extraction entirely
- Introduces generative models for entity extraction
- Modifies unrelated pipeline components

STOP CONDITION:
If the requested improvement cannot be made within the above constraints,
DO NOT implement a workaround. Instead, explain why and stop.

DEFAULT BIAS:
When in doubt, suppress diagnosis output.

## Project Overview
Medical NLP system for analyzing physician-patient conversations. Extracts medical entities, generates SOAP notes, and analyzes sentiment/intent using hybrid rule-based + transformer models (spaCy, BART, DistilBERT).

## Critical Architecture Patterns

### Multi-Mode NER Design (Hallucination Prevention)
The `MedicalNER` class implements **context-aware extraction** with two modes:
- **utterance mode**: Single patient statements - NEVER infers diagnosis without explicit confirmation ("The doctor said it was whiplash")
- **transcript mode**: Full conversations - requires clinical context + explicit mentions

**Key implementation** in [src/medical_ner.py](src/medical_ner.py#L145-L175):
```python
def extract_diagnosis(self, text: str, mode: str = "auto") -> str:
    if mode == "utterance":
        # STRICT: Only return if explicitly confirmed
        if self._has_explicit_diagnosis_confirmation(text_lower):
            return self._extract_confirmed_diagnosis(text_lower)
        return ""  # Prevent hallucination
```

**When modifying NER**: Always maintain conservative defaults - prefer empty strings over hallucinated data. See [test_diagnosis_hallucination.py](test_diagnosis_hallucination.py) for expected behavior.

### Pipeline Component Flow
Main entry: [main.py](main.py) → `MedicalTranscriptionPipeline` orchestrates 5 sequential steps:
1. **MedicalNER** - Rule-based + spaCy entity extraction
2. **KeywordExtractor** - Pattern matching for medical phrases  
3. **MedicalSummarizer** - BART-based summarization + JSON formatting
4. **SentimentIntentAnalyzer** - DistilBERT + fallback rules (Anxious/Neutral/Reassured)
5. **SOAPNoteGenerator** - Template-based clinical documentation

Each component is **stateless** and uses dict-based output. Results accumulate in a single JSON structure saved to `output/analysis_results.json`.

## Development Workflows

### Environment Setup (Windows PowerShell)
```powershell
.\setup.ps1  # Automated: venv, deps, spaCy model, output dir
```
**Manual alternative**:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Running & Testing
```powershell
python main.py                          # Process sample transcript
python examples.py                       # See multiple use cases
python tests.py                          # Component tests (6 functions)
python test_diagnosis_hallucination.py   # Hallucination prevention tests
```

**Testing convention**: Simple function-based tests (no pytest classes). Each `test_*()` function prints status and asserts conditions. See [tests.py](tests.py#L15-L145).

### Model Loading Patterns
All modules use **graceful degradation**:
```python
try:
    self.nlp = spacy.load("en_core_sci_sm")  # Preferred medical model
except:
    self.nlp = spacy.load("en_core_web_sm")  # Fallback to general
```
Transformers use `device=-1` (CPU) by default. Check [config.py](config.py) for model names.

## Project-Specific Conventions

### Configuration Structure
Central config in [config.py](config.py) with nested dicts:
- `MODELS`: Model identifiers for spaCy/transformers
- `*_CONFIG`: Per-component settings (NER, sentiment, SOAP, output)
- No environment variables - all hardcoded defaults

### Entity Extraction Patterns
Uses **triple strategy**:
1. Regex patterns for medical phrases (`r'(\d+\s+sessions?\s+of\s+physiotherapy)'`)
2. Keyword matching from predefined sets (`self.symptom_keywords`)
3. spaCy NER for person names/dates

Return format: `{"Symptoms": List[str], "Treatment": List[str], "Diagnosis": str, ...}`

**Capitalization rule**: Title case for symptoms/treatments (`"Neck pain"`, `"10 Physiotherapy Sessions"`)

### SOAP Note Generation
Template-based extraction from transcript + entities, NOT generative. Each section has dedicated logic:
- **Subjective**: Extract patient statements using regex on `patient:` lines
- **Objective**: Hardcoded patterns for exam findings
- **Assessment**: Maps directly to diagnosis entity
- **Plan**: Extracts treatment + prognosis

Output both JSON dict and formatted text string. See [src/soap_generator.py](src/soap_generator.py#L1-L100).

### Sentiment Analysis Three-Tier System
1. **Primary**: DistilBERT transformer (`distilbert-base-uncased-finetuned-sst-2-english`)
2. **Fallback**: Rule-based keyword matching if model fails
3. **Mapping**: POSITIVE→Reassured, NEGATIVE→Anxious, else Neutral

**Context-aware override**: Even if model says positive, check for anxiety keywords (`worried`, `concern`) and override to Anxious. Implemented in [src/sentiment_intent.py](src/sentiment_intent.py#L35-L62).

## Integration Points

### Data Flow Between Components
- **NER** outputs entities dict → consumed by Summarizer, SOAP Generator
- **Summarizer** needs entities for structured summary generation
- **SOAP Generator** requires both transcript (regex patterns) AND entities (data mapping)
- **Sentiment Analyzer** works independently on conversation utterances

**No component writes to disk** - only `main.py` saves final output.

### Sample Data Structure
Input from [data/sample_transcript.json](data/sample_transcript.json):
```json
{
  "transcript": "Physician: ... Patient: ..."
}
```

Output to [output/analysis_results.json](output/analysis_results.json):
```json
{
  "entities": {...},
  "keywords": [...],
  "structured_summary": "...",
  "sentiment_intent": [...],
  "soap_note": {...}
}
```

## Documentation References
- **Architecture details**: [ARCHITECTURE.md](ARCHITECTURE.md) - Component diagrams, data flow
- **API explanations**: [TECHNICAL_QA.md](TECHNICAL_QA.md) - Design decisions, edge cases
- **Quick commands**: [QUICKSTART.md](QUICKSTART.md) - Setup steps, usage examples
- **Example usage**: [examples.py](examples.py) - Practical demonstrations
- **Jupyter walkthrough**: [Medical_NLP_Pipeline.ipynb](Medical_NLP_Pipeline.ipynb) - Interactive exploration

## Key Files to Understand
- [src/medical_ner.py](src/medical_ner.py) (439 lines) - Core extraction logic, hallucination prevention
- [main.py](main.py) (224 lines) - Pipeline orchestration, output formatting
- [config.py](config.py) - All configuration constants
- [test_diagnosis_hallucination.py](test_diagnosis_hallucination.py) - Critical behavior tests for diagnosis extraction

## Anti-Patterns to Avoid
❌ Don't use generative models for entity extraction - this codebase is **rule-based + transformers for classification only**  
❌ Don't modify extraction logic without running `test_diagnosis_hallucination.py` - hallucination prevention is a core requirement  
❌ Don't assume GPU availability - all models configured for CPU (`device=-1`)  
❌ Don't add async/parallel processing - pipeline is intentionally sequential for simplicity  
❌ Don't create new config files - extend [config.py](config.py) dicts instead
