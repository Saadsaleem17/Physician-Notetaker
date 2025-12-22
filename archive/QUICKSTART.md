# Quick Start Guide

## Medical Transcription NLP Pipeline

### 🚀 Quick Setup (5 minutes)

#### Option 1: Automated Setup (Recommended)

```powershell
# Run the setup script
.\setup.ps1
```

#### Option 2: Manual Setup

```powershell
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Create output directory
New-Item -ItemType Directory -Force -Path "output"
```

---

### 📊 Run the Application

#### Quick Demo

```powershell
python main.py
```

This will:
- Process the sample physician-patient transcript
- Extract medical entities
- Analyze sentiment and intent
- Generate a SOAP note
- Save results to `output/analysis_results.json`

#### Run Examples

```powershell
python examples.py
```

Shows various use cases:
- Entity extraction
- Sentiment analysis
- Comparing multiple cases
- Batch processing

#### Run Tests

```powershell
python tests.py
```

Validates all components are working correctly.

---

### 📓 Interactive Jupyter Notebook

```powershell
jupyter notebook Medical_NLP_Pipeline.ipynb
```

The notebook provides:
- Step-by-step explanations
- Interactive examples
- Visualizations
- Q&A section

---

### 💻 Use in Your Code

```python
from src.medical_ner import MedicalNER
from src.sentiment_intent import SentimentIntentAnalyzer
from src.soap_generator import SOAPNoteGenerator

# Your transcript
transcript = """
Doctor: How are you feeling?
Patient: I had an accident. My neck hurts.
"""

# Extract entities
ner = MedicalNER()
entities = ner.extract_medical_entities(transcript)
print(entities)

# Analyze sentiment
analyzer = SentimentIntentAnalyzer()
result = analyzer.analyze("I'm worried about my pain")
print(f"Sentiment: {result['Sentiment']}")
print(f"Intent: {result['Intent']}")

# Generate SOAP note
soap_gen = SOAPNoteGenerator()
soap_note = soap_gen.generate_soap_note(transcript, entities)
print(soap_note)
```

---

### 📁 Expected Output

After running `main.py`, you'll get:

1. **Console Output**: All analysis results
2. **JSON File**: `output/analysis_results.json`

Example JSON structure:
```json
{
  "entities": {
    "Patient_Name": "Janet Jones",
    "Symptoms": ["Neck pain", "Back pain"],
    "Diagnosis": "Whiplash injury",
    "Treatment": ["10 physiotherapy sessions"],
    "Current_Status": "Occasional backache",
    "Prognosis": "Full recovery expected"
  },
  "sentiment_intent": [
    {
      "Text": "I'm doing better...",
      "Sentiment": "Neutral",
      "Intent": "Reporting symptoms"
    }
  ],
  "soap_note": {
    "Subjective": {...},
    "Objective": {...},
    "Assessment": {...},
    "Plan": {...}
  }
}
```

---

### 🔍 What Each Component Does

| Component | Function | Input | Output |
|-----------|----------|-------|--------|
| **Medical NER** | Extract medical info | Transcript | Symptoms, treatments, diagnosis |
| **Keyword Extraction** | Find key phrases | Transcript | List of medical terms |
| **Summarization** | Create summary | Transcript + Entities | Structured report |
| **Sentiment Analysis** | Detect emotions | Patient text | Anxious/Neutral/Reassured |
| **Intent Detection** | Understand goal | Patient text | Intent category |
| **SOAP Generator** | Clinical docs | Transcript + Entities | SOAP note |

---

### 🛠️ Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'transformers'`
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: `Can't find model 'en_core_web_sm'`
- **Solution**: Run `python -m spacy download en_core_web_sm`

**Issue**: Slow first run
- **Reason**: Downloading transformer models (one-time)
- **Solution**: Be patient, subsequent runs are faster

**Issue**: Out of memory
- **Solution**: Models use CPU by default. For large batches, process in chunks.

---

### 📚 Next Steps

1. **Explore the Notebook**: Best way to learn the system
2. **Try Your Own Data**: Replace sample transcript
3. **Read the README**: Comprehensive documentation
4. **Check Examples**: See different use cases
5. **Run Tests**: Ensure everything works

---

### 🎯 Key Features Demonstrated

✅ Named Entity Recognition (NER)  
✅ Medical keyword extraction  
✅ Text summarization  
✅ Sentiment analysis (Anxious/Neutral/Reassured)  
✅ Intent detection  
✅ SOAP note generation  
✅ JSON & text output formats  

---

### 📞 Need Help?

- Check `README.md` for detailed documentation
- Review code comments in `src/` modules
- Run `python tests.py` to diagnose issues
- Examine examples in `examples.py`

---

**Ready to start? Run:**

```powershell
python main.py
```

Enjoy! 🎉
