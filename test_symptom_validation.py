from src.medical_ner import MedicalNER

ner = MedicalNER()
text = "I have neck pain and back pain, but no anxiety or emotional issues."

print("=" * 70)
print("SYMPTOM VALIDATION TEST")
print("=" * 70)
print(f"\nInput: {text}\n")

# Test extraction
symptoms = ner.extract_symptoms(text, use_transformer=False)
print(f"Extracted symptoms (rule-based): {symptoms}")

# Test is_valid_symptom
print("\nValidation checks:")
print(f"  'Neck Pain' valid: {ner.is_valid_symptom('Neck Pain', text)}")
print(f"  'Back Pain' valid: {ner.is_valid_symptom('Back Pain', text)}")
print(f"  'Anxiety' valid: {ner.is_valid_symptom('Anxiety', text)}")
print(f"  'Pain' valid: {ner.is_valid_symptom('Pain', text)}")

# Test full entity extraction
entities = ner.extract_medical_entities(text, mode='utterance')
print(f"\nFinal symptoms after all filters: {entities['Symptoms']}")
