"""
Compare Rule-Based vs Transformer-Based NER
Test which approach extracts symptoms better
"""

from src.medical_ner import MedicalNER
import json

print("="*70)
print("NER COMPARISON: Rule-Based vs Transformer-Based")
print("="*70)

# Initialize NER
ner = MedicalNER()

# Test cases
test_cases = [
    "I am having cough doctor my lungs feel heavy and I am having headaches",
    "I have severe chest pain and shortness of breath",
    "My patient has fever, nausea, and vomiting since yesterday",
    "I'm experiencing dizziness, fatigue, and loss of appetite",
    "The pain in my abdomen is unbearable, and I have been vomiting",
    "I have a persistent cough, runny nose, and sore throat",
]

for i, text in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"TEST CASE {i}")
    print(f"{'='*70}")
    print(f"Input: \"{text}\"")
    print()
    
    # Get comparison results
    results = ner.extract_symptoms_hybrid(text)
    
    print("📋 RULE-BASED EXTRACTION:")
    if results['rule_based']:
        for symptom in results['rule_based']:
            print(f"   ✓ {symptom}")
    else:
        print("   (none)")
    
    print("\n🤖 TRANSFORMER-BASED EXTRACTION:")
    if results['transformer_based']:
        for symptom in results['transformer_based']:
            print(f"   ✓ {symptom}")
    else:
        print("   (none)")
    
    print("\n🔄 COMBINED (Best of Both):")
    if results['combined']:
        for symptom in results['combined']:
            print(f"   ✓ {symptom}")
    else:
        print("   (none)")
    
    # Show unique findings
    if results['rule_only']:
        print("\n📌 Found ONLY by Rule-Based:")
        for symptom in results['rule_only']:
            print(f"   → {symptom}")
    
    if results['transformer_only']:
        print("\n📌 Found ONLY by Transformer:")
        for symptom in results['transformer_only']:
            print(f"   → {symptom}")
    
    print(f"\n📊 Stats:")
    print(f"   Rule-based: {len(results['rule_based'])} symptoms")
    print(f"   Transformer: {len(results['transformer_based'])} symptoms")
    print(f"   Combined: {len(results['combined'])} symptoms")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)
print("\n💡 Recommendation: Use the method that catches more symptoms")
print("   or combine both for maximum coverage!")
