"""
Medical Transcription NLP Pipeline

A production-grade NLP system for analyzing medical consultation transcripts
and generating structured clinical documentation (SOAP notes).

Features:
- Hybrid Named Entity Recognition (NER) with transformer models
- Diagnosis inference with confidence scoring
- Sentiment and intent analysis
- Automated SOAP note generation
"""

__version__ = "1.0.0"
__author__ = "Saad Salim"

from .medical_ner import MedicalNER, KeywordExtractor, MedicalSummarizer
from .sentiment_intent import SentimentIntentAnalyzer, SentimentAnalyzer, IntentDetector
from .soap_generator import SOAPNoteGenerator

__all__ = [
    'MedicalNER',
    'KeywordExtractor',
    'MedicalSummarizer',
    'SentimentIntentAnalyzer',
    'SentimentAnalyzer',
    'IntentDetector',
    'SOAPNoteGenerator',
]

