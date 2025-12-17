"""
Configuration settings for Medical NLP Pipeline
"""

# Model Configuration
MODELS = {
    'spacy': 'en_core_web_sm',  # or 'en_core_sci_sm' for medical
    'summarization': 'facebook/bart-large-cnn',
    'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
}

# NER Configuration
NER_CONFIG = {
    'extract_patient_name': True,
    'extract_dates': True,
    'use_medical_keywords': True,
}

# Sentiment Analysis Configuration
SENTIMENT_CONFIG = {
    'labels': {
        'POSITIVE': 'Reassured',
        'NEGATIVE': 'Anxious',
        'NEUTRAL': 'Neutral'
    },
    'confidence_threshold': 0.7,
    'use_fallback_rules': True,
}

# Intent Detection Configuration
INTENT_CONFIG = {
    'categories': [
        'Seeking reassurance',
        'Reporting symptoms',
        'Describing treatment history',
        'Expressing concern',
        'Asking questions',
        'Expressing gratitude',
        'Confirming understanding',
        'General communication'
    ]
}

# SOAP Note Configuration
SOAP_CONFIG = {
    'include_patient_name': True,
    'include_date': True,
    'format': 'json',  # 'json' or 'text'
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_json': True,
    'save_text': True,
    'output_dir': 'output',
}

# Processing Configuration
PROCESSING_CONFIG = {
    'verbose': True,
    'save_intermediate': False,
}
