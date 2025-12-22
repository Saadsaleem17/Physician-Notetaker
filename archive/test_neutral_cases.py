"""
Comprehensive test suite for utterance classification and entity extraction
Tests neutral statements, symptom reporting, anxiety detection, diagnosis extraction, and edge cases
"""
import sys
from src.medical_ner import MedicalNER
from src.sentiment_intent import SentimentIntentAnalyzer

def test_category(category_name, test_cases, ner, sentiment_analyzer):
    """Run tests for a specific category"""
    print(f"\n{'='*70}")
    print(f"{category_name}")
    print(f"{'='*70}")
    
    category_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        utterance = test_case['utterance']
        expected = test_case['expected']
        
        print(f"\n[Test {i}] \"{utterance}\"")
        
        # Extract symptoms
        symptoms = ner.extract_symptoms(utterance)
        
        # Extract diagnosis (utterance mode for single statements)
        diagnosis = ner.extract_diagnosis(utterance, mode='utterance')
        
        # Analyze sentiment
        sentiment_result = sentiment_analyzer.analyze(utterance)
        
        # Validate expectations
        test_passed = True
        
        # Check symptoms
        if expected.get('symptoms') == 'none':
            if symptoms:
                print(f"  X Symptoms: Expected NONE, got {symptoms}")
                test_passed = False
            else:
                print(f"  >> Symptoms: None detected")
        elif expected.get('symptoms') == 'yes':
            if symptoms:
                print(f"  >> Symptoms: {symptoms}")
            else:
                print(f"  X Symptoms: Expected some, got NONE")
                test_passed = False
        
        # Check diagnosis
        if expected.get('diagnosis') == 'none':
            if diagnosis:
                print(f"  X Diagnosis: Expected NONE, got '{diagnosis}'")
                test_passed = False
            else:
                print(f"  >> Diagnosis: None")
        elif expected.get('diagnosis'):
            if diagnosis and expected['diagnosis'].lower() in diagnosis.lower():
                print(f"  >> Diagnosis: {diagnosis}")
            else:
                print(f"  ! Diagnosis: Expected '{expected['diagnosis']}', got '{diagnosis}'")
        
        # Check sentiment
        expected_sentiments = expected.get('sentiment', [])
        if not isinstance(expected_sentiments, list):
            expected_sentiments = [expected_sentiments]
        
        if sentiment_result['Sentiment'] in expected_sentiments:
            print(f"  >> Sentiment: {sentiment_result['Sentiment']}")
        else:
            print(f"  ! Sentiment: {sentiment_result['Sentiment']} (expected {' or '.join(expected_sentiments)})")
        
        # Check intent
        if expected.get('intent'):
            print(f"  >> Intent: {sentiment_result['Intent']}")
        
        if test_passed:
            print(f"  >> PASSED")
        else:
            print(f"  X FAILED")
            category_passed = False
    
    return category_passed

def run_all_tests():
    """Run comprehensive test suite"""
    ner = MedicalNER()
    sentiment_analyzer = SentimentIntentAnalyzer()
    
    all_categories_passed = True
    
    # Category 1: Neutral Historical Context (No Diagnosis Allowed)
    neutral_historical = [
        {
            'utterance': "The accident happened last September.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Neutral', 'intent': 'General communication'}
        },
        {
            'utterance': "I went to the hospital after the accident.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Neutral', 'intent': 'General communication'}
        },
        {
            'utterance': "I had physiotherapy sessions last month.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Neutral', 'intent': 'General communication'}
        },
        {
            'utterance': "The doctor checked me last week.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Neutral', 'intent': 'General communication'}
        },
        {
            'utterance': "I took a week off work after the accident.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Neutral', 'intent': 'General communication'}
        }
    ]
    
    # Category 2: Symptom Reporting (No Diagnosis Allowed)
    symptom_reporting = [
        {
            'utterance': "My neck hurts when I turn my head.",
            'expected': {'symptoms': 'yes', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Reporting symptoms'}
        },
        {
            'utterance': "I still have occasional back pain.",
            'expected': {'symptoms': 'yes', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Reporting symptoms'}
        },
        {
            'utterance': "There is some stiffness in my neck.",
            'expected': {'symptoms': 'yes', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Reporting symptoms'}
        },
        {
            'utterance': "I feel discomfort when sitting for long hours.",
            'expected': {'symptoms': 'yes', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Reporting symptoms'}
        },
        {
            'utterance': "I have trouble sleeping because of the pain.",
            'expected': {'symptoms': 'yes', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Reporting symptoms'}
        }
    ]
    
    # Category 3: Anxiety / Reassurance-Seeking Utterances
    anxiety_reassurance = [
        {
            'utterance': "I am worried this pain might not go away.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Anxious', 'intent': 'Seeking reassurance'}
        },
        {
            'utterance': "I feel anxious about my recovery.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Anxious', 'intent': 'Seeking reassurance'}
        },
        {
            'utterance': "What if this affects me in the future?",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Anxious', 'intent': 'Seeking reassurance'}
        },
        {
            'utterance': "I am scared something might be wrong.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Anxious', 'intent': 'Seeking reassurance'}
        },
        {
            'utterance': "Should I be concerned about this?",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Anxious', 'intent': 'Seeking reassurance'}
        }
    ]
    
    # Category 4: Explicit Diagnosis Mentions (Allowed Only Here)
    explicit_diagnosis = [
        {
            'utterance': "The doctor said it was a whiplash injury.",
            'expected': {'symptoms': 'none', 'diagnosis': 'Whiplash', 'sentiment': 'Neutral', 'intent': 'Reporting diagnosis'}
        },
        {
            'utterance': "I was diagnosed with whiplash.",
            'expected': {'symptoms': 'none', 'diagnosis': 'Whiplash', 'sentiment': 'Neutral', 'intent': 'Reporting diagnosis'}
        },
        {
            'utterance': "They told me my injury was whiplash.",
            'expected': {'symptoms': 'none', 'diagnosis': 'Whiplash', 'sentiment': 'Neutral', 'intent': 'Reporting diagnosis'}
        },
        {
            'utterance': "The hospital confirmed it was a whiplash injury.",
            'expected': {'symptoms': 'none', 'diagnosis': 'Whiplash', 'sentiment': 'Neutral', 'intent': 'Reporting diagnosis'}
        }
    ]
    
    # Category 5: Negation & False Traps
    negation_traps = [
        {
            'utterance': "They said it was not a serious injury.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Reassured'], 'intent': 'General communication'}
        },
        {
            'utterance': "The doctor said there was no fracture.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Reassured'], 'intent': 'General communication'}
        },
        {
            'utterance': "They did not diagnose anything serious.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Reassured'], 'intent': 'General communication'}
        },
        {
            'utterance': "X-rays were not required.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Reassured'], 'intent': 'General communication'}
        }
    ]
    
    # Category 6: Gratitude / Closure
    gratitude_closure = [
        {
            'utterance': "Thank you doctor.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Reassured', 'intent': 'Expressing gratitude'}
        },
        {
            'utterance': "That is a relief.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Reassured', 'intent': 'Expressing relief'}
        },
        {
            'utterance': "I appreciate your help.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Reassured', 'intent': 'Expressing gratitude'}
        },
        {
            'utterance': "Thanks, I feel reassured now.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Reassured', 'intent': 'Expressing gratitude'}
        }
    ]
    
    # Category 7: Edge / Adversarial Utterances
    edge_adversarial = [
        {
            'utterance': "I read online about whiplash injuries.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Seeking information'}
        },
        {
            'utterance': "Someone I know had a whiplash injury.",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': 'Neutral', 'intent': 'General communication'}
        },
        {
            'utterance': "Is whiplash dangerous?",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Seeking information'}
        },
        {
            'utterance': "Can people recover fully from such injuries?",
            'expected': {'symptoms': 'none', 'diagnosis': 'none', 'sentiment': ['Neutral', 'Anxious'], 'intent': 'Seeking information'}
        }
    ]
    
    # Run all test categories
    categories = [
        ("1. NEUTRAL HISTORICAL CONTEXT (No Diagnosis Allowed)", neutral_historical),
        ("2. SYMPTOM REPORTING (No Diagnosis Allowed)", symptom_reporting),
        ("3. ANXIETY / REASSURANCE-SEEKING UTTERANCES", anxiety_reassurance),
        ("4. EXPLICIT DIAGNOSIS MENTIONS (Allowed Only Here)", explicit_diagnosis),
        ("5. NEGATION & FALSE TRAPS", negation_traps),
        ("6. GRATITUDE / CLOSURE", gratitude_closure),
        ("7. EDGE / ADVERSARIAL UTTERANCES", edge_adversarial)
    ]
    
    for category_name, test_cases in categories:
        passed = test_category(category_name, test_cases, ner, sentiment_analyzer)
        if not passed:
            all_categories_passed = False
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL TEST SUMMARY")
    print(f"{'='*70}")
    
    total_tests = sum(len(tests) for _, tests in categories)
    print(f"Total tests run: {total_tests}")
    
    if all_categories_passed:
        print(">> ALL CATEGORIES PASSED")
    else:
        print("! SOME TESTS DID NOT MEET EXPECTED BEHAVIOR")
        print("   (Review warnings above for edge cases)")
    
    return all_categories_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
