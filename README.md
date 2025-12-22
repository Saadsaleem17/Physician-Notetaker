# Medical Transcription NLP Pipeline

A production-grade Natural Language Processing pipeline for analyzing medical consultation transcripts and generating structured clinical documentation. This system extracts medical entities, performs sentiment analysis, and automatically generates SOAP notes from physician-patient conversations.

---

## Overview

This pipeline processes raw medical transcripts and produces:
- **Structured Medical Entities**: Symptoms, diagnoses, treatments, investigations, prognosis
- **SOAP Notes**: Clinical documentation in Subjective-Objective-Assessment-Plan format
- **Sentiment & Intent Analysis**: Patient communication patterns and emotional states
- **Medical Keyword Extraction**: Key clinical phrases and terminology
- **JSON Summary**: Machine-readable structured output for downstream systems

---

## Technical Approach

### Architecture

The pipeline uses a **hybrid NLP approach** combining rule-based extraction with transformer models:

1. **Named Entity Recognition (NER)**
   - **Hybrid Method**: Rule-based patterns + BioBERT transformer model
   - Extracts: Symptoms, Diagnosis, Treatment, Investigations, Prognosis, Current Status
   - **Symptom Validation**: Multi-stage filtering to prevent hallucinations
   - **Negation Handling**: Context-aware negation detection (e.g., "no X-ray" ≠ performed)
   - **Body-Part Scoped Extraction**: Symptoms linked to specific anatomical regions

2. **Diagnosis Inference Layer**
   - **Confidence Scoring**: Weighs physician statements, linguistic triggers, symptom compatibility
   - **Speaker-Aware**: Prioritizes physician diagnostic statements over patient mentions
   - **Evidence Tracking**: Returns source sentences and reasoning for auditability

3. **Investigation Status Tracking**
   - **Three-State Model**: Performed / Considered / Negated
   - **Clinical Accuracy**: Prevents "no X-ray" from appearing as "investigation performed"
   - **Context Preservation**: Distinguishes between "exam done" and "MRI discussed if symptoms worsen"

4. **SOAP Note Generation**
   - **Narrative Chief Complaint**: Converts symptom lists to clinical narratives
   - **Temporal HPI**: Separates historical vs. current symptoms
   - **Evidence-Based Objective**: Uses documented exam findings, not inferred observations
   - **Structured Assessment & Plan**: Diagnosis severity grading and follow-up recommendations

5. **Sentiment & Intent Analysis**
   - **DistilBERT Sentiment Model**: Fine-tuned for medical dialogue
   - **Intent Classification**: 8 categories (Answering questions, Reporting symptoms, Seeking reassurance, etc.)
   - **Pattern Priority**: Ordered rule matching to prevent misclassification
   - **Word Count Validation**: Prevents long symptom descriptions from being classified as short answers

### Key Design Principles

- **No Hallucinations**: Strict gating prevents extraction without explicit textual evidence
- **Clinical Accuracy**: Follows medical documentation standards (e.g., no speculative diagnoses)
- **Transparency**: All extracted entities traceable to source text
- **Modular Architecture**: Separate concerns (NER, SOAP, Sentiment) for maintainability

---

## Setup Instructions

---

## Setup Instructions

### Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package manager)
- **Virtual environment** (recommended)

### Installation Steps

1. **Navigate to the project directory**
   ```bash
   cd "C:\Users\ACER\Desktop\Physician Notetaker"
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   ```bash
   # Windows PowerShell
   venv\Scripts\activate
   
   # Windows Command Prompt
   venv\Scripts\activate.bat
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

   The transformer models will be downloaded automatically on first run (~500MB total).

---

## Usage

### Running the Pipeline

Execute the main pipeline script:

```bash
python main.py
```

By default, this processes `data/sample_transcript.json` and outputs results to `output/analysis_results.json`.

---

## Input Data Format

### Creating Your Transcript File

Create a JSON file in the `data/` directory with the following structure:

```json
{
  "transcript_full": "Physician: Good morning, Ms. Jones. How are you feeling today?\nPatient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.\nPhysician: I understand you were in a car accident last September.\nPatient: Yes, it was on September 1st, around 12:30 in the afternoon..."
}
```

### Format Requirements

- **Field name**: Must be `transcript_full`
- **Speaker labels**: Use `Physician:` and `Patient:` prefixes
- **Line breaks**: Use `\n` to separate utterances
- **Format**: Plain text dialogue

### Example Structure

```
Physician: [Question or statement]
Patient: [Response]
Physician: [Follow-up]
Patient: [Response]
...
```

---

## Processing Your Own Data

### Step 1: Create Your Transcript

Create a new JSON file in the `data/` directory:

```json
{
  "transcript_full": "Physician: What brings you in today?\nPatient: I've been having severe wrist pain after a fall last week.\nPhysician: Did you seek any treatment?\nPatient: Yes, I went to the emergency room. They took an X-ray and said there was no fracture, just a sprain.\nPhysician: Let me examine your wrist. Can you move it?\nPatient: It hurts when I try to rotate it.\nPhysician: Your wrist shows some swelling but good circulation. This appears to be a resolving soft tissue injury.\nPatient: Will it get better?\nPhysician: Yes, with rest and a brace, you should see significant improvement in 2-3 weeks."
}
```

### Step 2: Update main.py (Optional)

If you want to process a different file, modify [main.py](main.py#L173):

```python
# Change this line:
transcript_path = Path(__file__).parent / 'data' / 'my_transcript.json'
```

### Step 3: Run the Pipeline

```bash
python main.py
```

---

## Output Format

### Console Output

The pipeline displays:
1. **Processing Steps**: Real-time progress (Loading models, extracting entities, etc.)
2. **JSON Summary**: Structured medical entities
3. **Key Medical Phrases**: Extracted terminology
4. **Sentiment & Intent Analysis**: Per-utterance classification
5. **SOAP Note**: Formatted clinical documentation

### JSON Output File

Results are automatically saved to `output/analysis_results.json`:

```json
{
  "entities": {
    "Patient_Name": "Jones",
    "Date_of_Incident": "September 1st",
    "Onset": "Sudden",
    "Duration": "Four Weeks",
    "Symptoms": ["Back Pain", "Head Impact", "Neck Pain", "Sleep Difficulty"],
    "Diagnosis": "Whiplash Injury",
    "Treatment": ["10 Physiotherapy Sessions", "Ice", "Painkillers"],
    "Investigations": {
      "performed": ["Examination"],
      "considered": [],
      "negated": ["X-ray"]
    },
    "Current_Status": "Occasional backaches",
    "Prognosis": "Full recovery within six months of the accident"
  },
  "keywords": [
    "physical examination",
    "full recovery",
    "neck and back pain",
    "car accident",
    "range of movement",
    "painkillers",
    "emergency",
    "steering wheel",
    "whiplash injury"
  ],
  "json_summary": {
    "Patient_Name": "Jones",
    "Date_of_Incident": "September 1st",
    "Symptoms": ["Back Pain", "Head Impact", "Neck Pain", "Sleep Difficulty"],
    "Diagnosis": "Whiplash Injury",
    "Treatment": ["10 Physiotherapy Sessions", "Ice", "Painkillers"],
    "Current_Status": "Occasional backaches",
    "Prognosis": "Full recovery within six months of the accident"
  },
  "sentiment_intent": [
    {
      "Text": "Good morning, doctor.",
      "Sentiment": "Reassured",
      "Intent": "General communication"
    },
    {
      "Text": "Yes, it was on September 1st, around 12:30 in the afternoon.",
      "Sentiment": "Anxious",
      "Intent": "Answering questions"
    },
    {
      "Text": "It's not constant, but I do get occasional backaches.",
      "Sentiment": "Anxious",
      "Intent": "Reporting symptoms"
    }
  ],
  "soap_note": {
    "Subjective": {
      "Chief_Complaint": "Neck and back pain following a car accident",
      "History_of_Present_Illness": "Initially had Head Impact, Neck Pain, Sleep Difficulty for Four Weeks. Currently reports Back Pain."
    },
    "Objective": {
      "Physical_Exam": "Full range of movement in the neck. Full range of movement in the back. No tenderness.",
      "Observations": "Clinical examination performed. No X-rays obtained."
    },
    "Assessment": {
      "Diagnosis": "Whiplash Injury",
      "Severity": "Mild, improving"
    },
    "Plan": {
      "Treatment": "10 Physiotherapy Sessions, Ice, Painkillers.",
      "Follow_Up": "Return for review if symptoms worsen or fail to improve within six weeks."
    }
  }
}
```

---

## Project Structure

```
Physician Notetaker/
├── main.py                      # Main pipeline execution
├── requirements.txt             # Python dependencies
├── README.md                   # This file
├── data/
│   └── sample_transcript.json  # Example input data
├── output/
│   └── analysis_results.json   # Pipeline output (generated)
└── src/
    ├── __init__.py
    ├── medical_ner.py          # NER, diagnosis inference, summarization
    ├── sentiment_intent.py     # Sentiment & intent analysis
    └── soap_generator.py       # SOAP note generation
```

---

## Module Details

### [src/medical_ner.py](src/medical_ner.py)
- **MedicalNER**: Hybrid NER for symptoms, diagnosis, treatment, investigations
- **DiagnosisInferenceLayer**: Confidence-scored diagnosis extraction
- **KeywordExtractor**: Medical phrase extraction
- **MedicalSummarizer**: Structured text summarization

### [src/sentiment_intent.py](src/sentiment_intent.py)
- **SentimentAnalyzer**: DistilBERT-based emotion classification
- **IntentDetector**: 8-category intent classification
- **SentimentIntentAnalyzer**: Combined analysis per utterance

### [src/soap_generator.py](src/soap_generator.py)
- **SOAPNoteGenerator**: Converts entities to SOAP format
- Narrative chief complaints
- Temporal HPI construction
- Evidence-based objective findings

---

## Example Workflow

1. **Prepare your transcript**:
   - Record physician-patient dialogue
   - Format as JSON with `transcript_full` field
   - Use `Physician:` and `Patient:` speaker labels

2. **Run the pipeline**:
   ```bash
   python main.py
   ```

3. **Review outputs**:
   - Console: Human-readable SOAP note and analysis
   - JSON file: Machine-readable structured data in `output/`

4. **Integrate with downstream systems**:
   - Parse `output/analysis_results.json`
   - Extract entities, SOAP sections, or sentiment data
   - Use for EHR integration, analytics, or quality monitoring

---

## Performance Notes

- **Processing Time**: ~5-10 seconds per transcript (depends on length and hardware)
- **Model Loading**: First run downloads transformer models (~500MB), subsequent runs are faster
- **Accuracy**: Validated on medical consultation datasets with clinical expert review
- **Scalability**: Batch processing can be implemented by iterating over multiple JSON files

---

## Troubleshooting

### Common Issues

1. **"spaCy model not found"**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **"Transformer model download fails"**
   - Check internet connection
   - Models auto-download on first run
   - Requires ~500MB disk space

3. **"ModuleNotFoundError"**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

4. **"JSON decode error"**
   - Validate JSON syntax using a JSON validator
   - Ensure `transcript_full` field exists
   - Check for proper escape sequences (`\n` for newlines)

5. **"Out of memory"**
   - Reduce transcript length
   - Close other applications
   - Transformer models require ~4GB RAM

---

## Technical Requirements

**Minimum:**
- Python 3.8+
- 4GB RAM
- 2GB disk space (for models)
- Internet connection (first run only)

**Recommended:**
- Python 3.10+
- 8GB RAM
- GPU (optional, for faster transformer inference)

---

**Last Updated**: December 22, 2025

For questions or issues:
- Open an issue on GitHub
- Check documentation in Jupyter notebook
- Review code comments in source files

---

**Note**: This system is for educational and research purposes. Not intended for clinical use without proper validation and regulatory compliance.
