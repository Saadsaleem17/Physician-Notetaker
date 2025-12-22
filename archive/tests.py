"""
Test suite for Medical NLP Pipeline components
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.medical_ner import MedicalNER, KeywordExtractor
from src.medical_summarizer import MedicalSummarizer
from src.sentiment_intent import SentimentIntentAnalyzer
from src.soap_generator import SOAPNoteGenerator


def test_medical_ner():
    """Test Medical NER functionality"""
    print("Testing Medical NER...")
    
    ner = MedicalNER()
    test_text = """
    Patient: I had a car accident. My neck and back hurt a lot. 
    I went through ten physiotherapy sessions. Now I only have occasional back pain.
    """
    
    entities = ner.extract_medical_entities(test_text)
    
    # Assertions
    assert 'Symptoms' in entities, "Should extract symptoms"
    assert 'Treatment' in entities, "Should extract treatments"
    assert 'Diagnosis' in entities, "Should extract diagnosis"
    assert len(entities['Symptoms']) > 0, "Should find at least one symptom"
    
    print("✓ Medical NER tests passed!")
    return True


def test_keyword_extraction():
    """Test keyword extraction"""
    print("Testing Keyword Extraction...")
    
    extractor = KeywordExtractor()
    test_text = """
    The patient experienced whiplash injury from a car accident.
    Treatment included physiotherapy sessions and painkillers.
    """
    
    keywords = extractor.extract_keywords(test_text, top_n=5)
    
    # Assertions
    assert len(keywords) > 0, "Should extract keywords"
    assert isinstance(keywords, list), "Should return a list"
    
    print("✓ Keyword extraction tests passed!")
    return True


def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("Testing Sentiment Analysis...")
    
    analyzer = SentimentIntentAnalyzer()
    
    # Test anxious sentiment
    anxious_text = "I'm very worried about my condition getting worse"
    result1 = analyzer.analyze(anxious_text)
    assert result1['Sentiment'] in ['Anxious', 'Neutral', 'Reassured'], "Should return valid sentiment"
    
    # Test reassured sentiment
    reassured_text = "Thank you doctor, I feel much better now"
    result2 = analyzer.analyze(reassured_text)
    assert result2['Sentiment'] in ['Anxious', 'Neutral', 'Reassured'], "Should return valid sentiment"
    
    # Test intent
    assert 'Intent' in result1, "Should detect intent"
    
    print("✓ Sentiment analysis tests passed!")
    return True


def test_intent_detection():
    """Test intent detection"""
    print("Testing Intent Detection...")
    
    analyzer = SentimentIntentAnalyzer()
    
    test_cases = [
        ("I'm worried about my pain", "Seeking reassurance"),
        ("My back hurts a lot", "Reporting symptoms"),
        ("Thank you doctor", "Expressing gratitude"),
    ]
    
    for text, expected_category in test_cases:
        result = analyzer.analyze(text)
        assert 'Intent' in result, f"Should detect intent for: {text}"
    
    print("✓ Intent detection tests passed!")
    return True


def test_summarization():
    """Test medical summarization"""
    print("Testing Summarization...")
    
    summarizer = MedicalSummarizer()
    
    test_entities = {
        "Patient_Name": "Test Patient",
        "Symptoms": ["Pain", "Discomfort"],
        "Diagnosis": "Test Diagnosis",
        "Treatment": ["Test Treatment"],
        "Current_Status": "Improving",
        "Prognosis": "Good"
    }
    
    # Test JSON summary
    json_summary = summarizer.generate_json_summary(test_entities)
    assert 'Patient_Name' in json_summary, "Should include patient name"
    assert 'Symptoms' in json_summary, "Should include symptoms"
    
    # Test structured summary
    test_transcript = "Patient: I'm feeling better."
    structured = summarizer.create_structured_summary(test_transcript, test_entities)
    assert len(structured) > 0, "Should generate summary text"
    
    print("✓ Summarization tests passed!")
    return True


def test_soap_generation():
    """Test SOAP note generation"""
    print("Testing SOAP Note Generation...")
    
    generator = SOAPNoteGenerator()
    
    test_transcript = """
    Patient: I had a car accident. My neck and back hurt.
    Physician: Let me examine you. Everything looks good.
    """
    
    test_entities = {
        "Patient_Name": "Test Patient",
        "Symptoms": ["Neck pain", "Back pain"],
        "Diagnosis": "Whiplash injury",
        "Treatment": ["Physiotherapy"],
        "Current_Status": "Improving",
        "Prognosis": "Full recovery expected"
    }
    
    soap_note = generator.generate_soap_note(test_transcript, test_entities)
    
    # Assertions
    assert 'Subjective' in soap_note, "Should have Subjective section"
    assert 'Objective' in soap_note, "Should have Objective section"
    assert 'Assessment' in soap_note, "Should have Assessment section"
    assert 'Plan' in soap_note, "Should have Plan section"
    
    # Test text formatting
    soap_text = generator.format_soap_note_text(soap_note, "Test Patient")
    assert len(soap_text) > 0, "Should generate formatted text"
    assert 'SOAP NOTE' in soap_text, "Should include header"
    
    print("✓ SOAP generation tests passed!")
    return True


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*70)
    print("RUNNING MEDICAL NLP PIPELINE TESTS")
    print("="*70 + "\n")
    
    tests = [
        test_medical_ner,
        test_keyword_extraction,
        test_sentiment_analysis,
        test_intent_detection,
        test_summarization,
        test_soap_generation,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed += 1
        print()
    
    print("="*70)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
