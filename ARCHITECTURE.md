# System Architecture Diagram

## Medical Transcription NLP Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Physician-Patient Conversation Transcript                          │
│  ┌────────────────────────────────────────────────────────┐        │
│  │ "Doctor: How are you feeling today?"                   │        │
│  │ "Patient: I had a car accident. My neck hurts..."      │        │
│  └────────────────────────────────────────────────────────┘        │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  1. MEDICAL NER (medical_ner.py)                           │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • spaCy + Rule-based extraction                           │   │
│  │  • Extract: Symptoms, Treatments, Diagnosis, Prognosis     │   │
│  │  • Handle: Missing data with fallbacks                     │   │
│  │                                                             │   │
│  │  Output: {                                                  │   │
│  │    "Symptoms": ["Neck pain", "Back pain"],                 │   │
│  │    "Treatment": ["Physiotherapy"],                         │   │
│  │    "Diagnosis": "Whiplash injury"                          │   │
│  │  }                                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  2. KEYWORD EXTRACTION (medical_ner.py)                    │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Pattern-based extraction                                │   │
│  │  • Medical phrase identification                           │   │
│  │                                                             │   │
│  │  Output: ["whiplash injury", "physiotherapy sessions"]     │   │
│  └────────────────────────────────────────────────────────────┘   │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  3. SUMMARIZATION (medical_summarizer.py)                  │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • BART-large-CNN (Transformer)                            │   │
│  │  • Structured text generation                              │   │
│  │  • JSON formatting                                         │   │
│  │                                                             │   │
│  │  Output: Structured medical summary                        │   │
│  └────────────────────────────────────────────────────────────┘   │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  4. SENTIMENT ANALYSIS (sentiment_intent.py)               │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • DistilBERT (Transformer)                                │   │
│  │  • 3-class: Anxious/Neutral/Reassured                      │   │
│  │  • Rule-based fallback                                     │   │
│  │                                                             │   │
│  │  Output: {                                                  │   │
│  │    "Sentiment": "Anxious",                                 │   │
│  │    "Confidence": 0.87                                      │   │
│  │  }                                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  5. INTENT DETECTION (sentiment_intent.py)                 │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Pattern matching                                        │   │
│  │  • Context-aware classification                            │   │
│  │  • 8 intent categories                                     │   │
│  │                                                             │   │
│  │  Output: {                                                  │   │
│  │    "Intent": "Seeking reassurance"                         │   │
│  │  }                                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                               ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐   │
│  │  6. SOAP NOTE GENERATION (soap_generator.py)               │   │
│  │  ─────────────────────────────────────────────────────────  │   │
│  │  • Template-based extraction                               │   │
│  │  • Section mapping (S/O/A/P)                               │   │
│  │  • Clinical formatting                                     │   │
│  │                                                             │   │
│  │  Output: {                                                  │   │
│  │    "Subjective": {...},                                    │   │
│  │    "Objective": {...},                                     │   │
│  │    "Assessment": {...},                                    │   │
│  │    "Plan": {...}                                           │   │
│  │  }                                                          │   │
│  └────────────────────────────────────────────────────────────┘   │
│                                                                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐ │
│  │   JSON Format    │  │   Text Format    │  │  Console Display │ │
│  │  ──────────────  │  │  ──────────────  │  │  ──────────────  │ │
│  │ {                │  │  MEDICAL         │  │  ✓ Entities      │ │
│  │   "entities": {} │  │  CONSULTATION    │  │  ✓ Sentiment     │ │
│  │   "sentiment": {}│  │  SUMMARY         │  │  ✓ SOAP Note     │ │
│  │   "soap": {}     │  │  Patient: ...    │  │  ✓ Keywords      │ │
│  │ }                │  │  Diagnosis: ...  │  │                  │ │
│  │                  │  │  ...             │  │                  │ │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘ │
│                                                                      │
│  Saved to: output/analysis_results.json                             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Dependencies

```
main.py
  │
  ├── src/medical_ner.py
  │   ├── spacy
  │   └── regex patterns
  │
  ├── src/medical_summarizer.py
  │   ├── transformers (BART)
  │   └── medical_ner (for entities)
  │
  ├── src/sentiment_intent.py
  │   ├── transformers (DistilBERT)
  │   └── rule-based patterns
  │
  └── src/soap_generator.py
      ├── medical_ner (for entities)
      └── template-based extraction
```

## Data Flow

```
Transcript Text
      │
      ├─► Medical NER ──────────┐
      │                         │
      ├─► Keyword Extract ──────┼─► Entities Dictionary
      │                         │
      ├─► Summarizer ───────────┘
      │
      ├─► Sentiment Analyzer ──────► Sentiment + Intent
      │
      └─► SOAP Generator ──────────► SOAP Note
                                      │
                                      ▼
                              Combined Output
                                      │
                                      ▼
                              JSON + Text Files
```

## Model Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    MODEL STACK                            │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  NER Layer                                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │ spaCy (en_core_web_sm / en_core_sci_sm)           │  │
│  │ + Custom Rule-based Extractors                     │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  Summarization Layer                                      │
│  ┌────────────────────────────────────────────────────┐  │
│  │ BART-large-CNN (facebook/bart-large-cnn)           │  │
│  │ Seq2Seq Transformer (1024 hidden units)            │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  Sentiment Layer                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ DistilBERT (distilbert-base-uncased-finetuned)    │  │
│  │ 6-layer Transformer (768 hidden units)             │  │
│  │ + Rule-based Fallback                              │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  Intent Detection Layer                                   │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Pattern Matching + Context Analysis                │  │
│  │ Regex-based Classification                          │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  SOAP Generation Layer                                    │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Template-based Section Extraction                  │  │
│  │ Rule-based Mapping to S/O/A/P                      │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

## Hybrid Approach

```
┌─────────────────────────────────────────────────────────┐
│              HYBRID NLP PIPELINE                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Rule-Based Components (Fast, Interpretable)            │
│  ┌────────────────────────────────────────────────┐    │
│  │ • Pattern matching for medical terms           │    │
│  │ • Keyword extraction                            │    │
│  │ • SOAP template mapping                         │    │
│  │ • Fallback sentiment analysis                   │    │
│  └────────────────────────────────────────────────┘    │
│                         +                               │
│  ML-Based Components (Accurate, Flexible)               │
│  ┌────────────────────────────────────────────────┐    │
│  │ • Transformer models (BART, DistilBERT)        │    │
│  │ • spaCy NER                                     │    │
│  │ • Context understanding                         │    │
│  │ • Semantic analysis                             │    │
│  └────────────────────────────────────────────────┘    │
│                         =                               │
│  Robust, Production-Ready System                        │
│  ┌────────────────────────────────────────────────┐    │
│  │ ✓ Handles missing data                         │    │
│  │ ✓ Fast processing                               │    │
│  │ ✓ Interpretable results                         │    │
│  │ ✓ Extensible architecture                       │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Usage Flow

```
User
 │
 ├─► Option 1: Run main.py
 │      │
 │      ├─► Load transcript
 │      ├─► Initialize all components
 │      ├─► Process through pipeline
 │      ├─► Display results
 │      └─► Save to JSON
 │
 ├─► Option 2: Jupyter Notebook
 │      │
 │      ├─► Interactive cells
 │      ├─► Step-by-step execution
 │      ├─► Visual outputs
 │      └─► Experimentation
 │
 ├─► Option 3: Import as Module
 │      │
 │      ├─► from src import *
 │      ├─► Custom processing
 │      └─► Integration with other systems
 │
 └─► Option 4: Run examples.py
        │
        ├─► Predefined use cases
        ├─► Multiple scenarios
        └─► Batch processing demos
```

---

**Legend:**
- `┌─┐` = Component/Module
- `│` = Data flow
- `▼` = Processing direction
- `►` = User interaction
- `+` = Combination
- `=` = Result
