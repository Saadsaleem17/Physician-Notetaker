"""
Test suite for Medical NLP Pipeline
Focus:
- Sentiment accuracy
- Intent sanity
- Diagnosis hallucination prevention
"""

from src.sentiment_intent import SentimentIntentAnalyzer
from src.medical_ner import MedicalNER

sentiment_analyzer = SentimentIntentAnalyzer()
ner = MedicalNER()

# -----------------------------
# Test cases
# -----------------------------

TEST_CASES = [
    {
        "text": "I am feeling so good and better now",
        "expected_sentiment": "Reassured",
        "expected_intent": "Expressing improvement",
        "expect_symptoms": False,
        "expect_diagnosis": False
    },
    {
        "text": "My neck and back hurt a lot after the accident",
        "expected_sentiment": "Anxious",
        "expected_intent": "Reporting symptoms",
        "expect_symptoms": True,
        "expect_diagnosis": False
    },
    {
        "text": "I still get occasional back pain but it is improving",
        "expected_sentiment": "Neutral",
        "expected_intent": "Reporting symptoms",
        "expect_symptoms": True,
        "expect_diagnosis": False
    },
    {
        "text": "I am worried this pain will never go away",
        "expected_sentiment": "Anxious",
        "expected_intent": "Seeking reassurance",
        "expect_symptoms": True,
        "expect_diagnosis": False
    },
    {
        "text": "The doctor said it was a whiplash injury",
        "expected_sentiment": "Neutral",
        "expected_intent": "Reporting diagnosis",
        "expect_symptoms": False,
        "expect_diagnosis": True
    },
    {
        "text": "Thank you doctor, that is a relief",
        "expected_sentiment": "Reassured",
        "expected_intent": "Expressing gratitude",
        "expect_symptoms": False,
        "expect_diagnosis": False
    }
]

# -----------------------------
# Evaluation
# -----------------------------

def run_tests():
    sentiment_correct = 0
    intent_correct = 0
    diagnosis_errors = 0

    print("=" * 70)
    print("RUNNING MEDICAL NLP TEST SUITE")
    print("=" * 70)

    for idx, case in enumerate(TEST_CASES, 1):
        print(f"\nTest Case {idx}")
        print(f"Text: {case['text']}")

        sentiment_result = sentiment_analyzer.analyze(case["text"])
        entities = ner.extract_medical_entities(case["text"])

        # Sentiment check
        sentiment_match = sentiment_result["Sentiment"] == case["expected_sentiment"]
        intent_match = sentiment_result["Intent"] == case["expected_intent"]

        if sentiment_match:
            sentiment_correct += 1
        if intent_match:
            intent_correct += 1

        # Diagnosis sanity check
        has_diagnosis = bool(entities.get("Diagnosis"))
        if has_diagnosis and not case["expect_diagnosis"]:
            diagnosis_errors += 1

        print(f"Predicted Sentiment: {sentiment_result['Sentiment']} | Expected: {case['expected_sentiment']}")
        print(f"Predicted Intent: {sentiment_result['Intent']} | Expected: {case['expected_intent']}")
        print(f"Diagnosis Extracted: {entities.get('Diagnosis')}")

    total = len(TEST_CASES)

    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)

    print(f"Sentiment Accuracy: {sentiment_correct}/{total} = {sentiment_correct / total:.2f}")
    print(f"Intent Accuracy: {intent_correct}/{total} = {intent_correct / total:.2f}")
    print(f"Diagnosis Hallucination Errors: {diagnosis_errors}")

    if diagnosis_errors > 0:
        print("\n⚠️ WARNING: Diagnosis hallucination detected. Review NER gating logic.")
    else:
        print("\n✓ No diagnosis hallucination detected.")

    print("=" * 70)


if __name__ == "__main__":
    run_tests()
