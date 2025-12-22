"""
Demonstrate the superiority of context-aware gating over lexical filtering.

This shows how the new approach handles edge cases that would break skip_words.
"""

from src.medical_ner import MedicalNER

def test_gating_robustness():
    """Test cases that would fail with skip_words but pass with context gating"""
    
    ner = MedicalNER()
    
    print("="*70)
    print("CONTEXT-AWARE GATING: Edge Case Handling")
    print("="*70)
    print("\nThese cases would require expanding skip_words endlessly.")
    print("With context gating, they work immediately.\n")
    
    # Cases that would fail with skip_words approach
    test_cases = [
        {
            "text": "I visited ER last week",  # "hospital" not in text
            "expected_diagnosis": None,
            "reason": "No 'hospital' keyword, but ER is synonym"
        },
        {
            "text": "I saw a clinician who checked me",  # "doctor" not in text
            "expected_diagnosis": None,
            "reason": "No 'doctor' keyword, but clinician is synonym"
        },
        {
            "text": "Physio helped me recover",  # "physiotherapy" abbreviated
            "expected_diagnosis": None,
            "reason": "Abbreviated form not in skip_words"
        },
        {
            "text": "It happened in Oct",  # "September" not in text
            "expected_diagnosis": None,
            "reason": "Different month abbreviation"
        },
        {
            "text": "Ten days ago I felt pain",  # Number variant
            "expected_diagnosis": None,
            "reason": "Number written as word vs digit"
        },
        {
            "text": "I have severe persistent headache",  # "severe" + "persistent" are meaningful
            "expected_diagnosis": None,
            "reason": "Skip_words would remove clinically meaningful terms"
        },
        {
            "text": "The physician diagnosed me with whiplash",  # Explicit confirmation
            "expected_diagnosis": "Whiplash",
            "reason": "Has diagnostic context - should extract"
        },
        {
            "text": "Emergency room confirmed it was a strain",  # Variant of "hospital confirmed"
            "expected_diagnosis": "Strain",  # Will extract "Strain" 
            "reason": "Diagnostic confirmation with ER instead of hospital"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        text = test["text"]
        expected = test["expected_diagnosis"]
        reason = test["reason"]
        
        result = ner.extract_diagnosis(text, mode='utterance')
        
        # Check if result matches expectation
        if expected is None:
            success = (result == "" or result is None)
        else:
            success = (result and expected.lower() in result.lower())
        
        print(f"[Test {i}] \"{text}\"")
        print(f"  Reason: {reason}")
        print(f"  Expected: {expected if expected else 'None'}")
        print(f"  Got: {result if result else 'None'}")
        
        if success:
            print(f"  >> PASSED")
            passed += 1
        else:
            print(f"  X FAILED")
            failed += 1
        print()
    
    print("="*70)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*70)
    
    if failed == 0:
        print("\n>> Context-aware gating handles all edge cases without lexical lists!")
    else:
        print(f"\n! {failed} cases need attention")

def demonstrate_semantic_reasoning():
    """Show how gating reasons about meaning, not words"""
    
    ner = MedicalNER()
    
    print("\n" + "="*70)
    print("SEMANTIC REASONING: Context Detection")
    print("="*70)
    print("\nSame words, different context = different decisions\n")
    
    cases = [
        ("I have pain", False, "No diagnostic context"),
        ("Doctor said I have pain", False, "Statement about symptom, not diagnosis"),
        ("Doctor said it was whiplash", True, "Explicit diagnosis: 'said it was'"),
        ("I was diagnosed with whiplash", True, "Explicit diagnosis: 'diagnosed with'"),
    ]
    
    for text, should_have_diagnosis, explanation in cases:
        has_context = ner.has_diagnostic_context(text)
        diagnosis = ner.extract_diagnosis(text, mode='utterance')
        
        print(f'Text: "{text}"')
        print(f'  Diagnostic context detected: {has_context}')
        print(f'  Diagnosis extracted: {diagnosis if diagnosis else "None"}')
        print(f'  Explanation: {explanation}')
        print()

if __name__ == "__main__":
    test_gating_robustness()
    demonstrate_semantic_reasoning()
