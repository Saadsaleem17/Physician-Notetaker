"""
Quick sentiment analysis for a patient utterance
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.sentiment_intent import SentimentIntentAnalyzer
    from src.medical_ner import MedicalNER
    
    # Get input from user
    print("="*70)
    print("MEDICAL NLP ANALYSIS")
    print("="*70)
    patient_text = input("\nEnter patient utterance: ")
    
    if not patient_text.strip():
        print("No input provided. Exiting.")
        sys.exit(0)
    
    print("\n" + "-" * 70)
    print("PROCESSING...")
    print("-" * 70)
    
    # Initialize NLP components
    sentiment_analyzer = SentimentIntentAnalyzer()
    ner = MedicalNER()
    
    # Run NLP Analysis
    sentiment_result = sentiment_analyzer.analyze(patient_text)
    entities = ner.extract_medical_entities(patient_text)
    
    # Display results (NLP-generated, not hardcoded)
    print(f"\n📝 INPUT: \"{patient_text}\"")
    print("\n" + "="*70)
    print("NLP ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\n💭 SENTIMENT: {sentiment_result['Sentiment']}")
    print(f"🎯 INTENT: {sentiment_result['Intent']}")
    
    if entities['Symptoms']:
        print(f"\n🩺 EXTRACTED SYMPTOMS: {', '.join(entities['Symptoms'])}")
    
    if entities['Diagnosis']:
        print(f"📋 DIAGNOSIS: {entities['Diagnosis']}")
    
    # NLP-based clinical assessment
    print("\n" + "="*70)
    print("CLINICAL ASSESSMENT (NLP-Generated)")
    print("="*70)
    
    if sentiment_result['Sentiment'] == 'Anxious':
        print("⚠️  Patient shows signs of anxiety/distress")
    elif sentiment_result['Sentiment'] == 'Reassured':
        print("✓ Patient appears reassured/positive")
    else:
        print("ℹ️  Patient sentiment is neutral")
    
    if 'pain' in patient_text.lower():
        print("⚠️  Pain symptoms detected - requires attention")
    
    if any(word in patient_text.lower() for word in ['survive', 'die', 'death', 'fatal']):
        print("🚨 URGENT: Life-threatening concerns expressed")
    
    print("="*70)
    
except ImportError:
    print("\n❌ ERROR: Required packages not installed yet.")
    print("\nPlease run setup first:")
    print("  .\\setup.ps1")
    print("\nOr install manually:")
    print("  pip install -r requirements.txt")
    print("  python -m spacy download en_core_web_sm")
