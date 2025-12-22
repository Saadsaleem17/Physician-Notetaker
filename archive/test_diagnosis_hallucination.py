"""
Diagnosis Hallucination Prevention Tests
Tests that the NER does NOT hallucinate diagnoses from context-free utterances
"""

from src.medical_ner import MedicalNER

print("="*70)
print("DIAGNOSIS HALLUCINATION PREVENTION TESTS")
print("="*70)

ner = MedicalNER()

# Test cases that should NOT produce diagnosis
no_diagnosis_cases = [
    "I am feeling so good and better now",
    "Thank you doctor",
    "I am worried about my pain",
    "My back hurts",
    "I feel great today",
    "I'm doing well",
    "That's a relief",
    "I appreciate your help",
    "How long will recovery take?",
    "I am in pain and I feel I won't survive"
]

# Test cases that SHOULD produce diagnosis (explicit confirmation)
yes_diagnosis_cases = [
    "The doctor said it was a whiplash injury",
    "I was diagnosed with whiplash",
    "The emergency room told me it was a whiplash injury",
    "They confirmed it was a lower back strain",
    "The medical report showed whiplash injury"
]

print("\n📋 Testing cases that should NOT extract diagnosis (utterance mode):")
print("-"*70)

hallucination_errors = 0
for text in no_diagnosis_cases:
    entities = ner.extract_medical_entities(text, mode="utterance")
    diagnosis = entities.get('Diagnosis', '')
    
    if diagnosis:
        hallucination_errors += 1
        print(f"❌ FAILED: \"{text}\"")
        print(f"   Hallucinated: \"{diagnosis}\"")
    else:
        print(f"✓ PASSED: \"{text}\" → No diagnosis")

print("\n📋 Testing cases that SHOULD extract diagnosis (explicit confirmation):")
print("-"*70)

missing_diagnosis_errors = 0
for text in yes_diagnosis_cases:
    entities = ner.extract_medical_entities(text, mode="utterance")
    diagnosis = entities.get('Diagnosis', '')
    
    if not diagnosis:
        missing_diagnosis_errors += 1
        print(f"❌ FAILED: \"{text}\"")
        print(f"   Should extract diagnosis but got: \"\"")
    else:
        print(f"✓ PASSED: \"{text}\" → \"{diagnosis}\"")

print("\n" + "="*70)
print("FULL TRANSCRIPT MODE TEST")
print("="*70)

transcript = """
Physician: Good morning, Ms. Jones. How are you feeling today?
Patient: I'm doing better, but I still have some discomfort now and then.
Physician: I understand you were in a car accident last September.
Patient: Yes, it was on September 1st. I had hit my head on the steering wheel, 
and I could feel pain in my neck and back almost right away.
Patient: The first four weeks were rough. My neck and back pain were really bad.
I had to go through ten sessions of physiotherapy.
Physician: Everything looks good. I'd expect you to make a full recovery within six months.
"""

print("\nTesting full transcript (should allow diagnosis extraction with context):")
entities_transcript = ner.extract_medical_entities(transcript, mode="transcript")
print(f"Diagnosis extracted: \"{entities_transcript.get('Diagnosis', '')}\"")

print("\n" + "="*70)
print("TEST RESULTS SUMMARY")
print("="*70)

total_no_diagnosis = len(no_diagnosis_cases)
total_yes_diagnosis = len(yes_diagnosis_cases)

print(f"\n✅ Hallucination Prevention:")
print(f"   {total_no_diagnosis - hallucination_errors}/{total_no_diagnosis} cases correctly suppressed diagnosis")
print(f"   Hallucination errors: {hallucination_errors}")

print(f"\n✅ Explicit Diagnosis Detection:")
print(f"   {total_yes_diagnosis - missing_diagnosis_errors}/{total_yes_diagnosis} cases correctly extracted diagnosis")
print(f"   Missing extractions: {missing_diagnosis_errors}")

if hallucination_errors == 0:
    print("\n🎉 SUCCESS: No diagnosis hallucination detected!")
else:
    print(f"\n⚠️  WARNING: {hallucination_errors} diagnosis hallucination errors found")

print("="*70)
