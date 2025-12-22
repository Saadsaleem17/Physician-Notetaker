from src.medical_ner import MedicalNER
import json

with open('data/sample_transcript.json') as f:
    text = json.load(f)['transcript_full']

ner = MedicalNER()

print("Testing diagnosis extraction:")
print(f"Has diagnostic context: {ner.has_diagnostic_context(text)}")
print(f"Diagnosis extracted: '{ner.extract_diagnosis(text)}'")

# Check specific patterns
text_lower = text.lower()
print(f"\n'just a sprain' in text: {'just a sprain' in text_lower}")
print(f"'soft tissue injury' in text: {'soft tissue injury' in text_lower}")
print(f"'resolving soft tissue injury' in text: {'resolving soft tissue injury' in text_lower}")
