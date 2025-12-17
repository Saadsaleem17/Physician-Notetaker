"""
Example usage demonstrating different components of the medical NLP pipeline
"""

import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.medical_ner import MedicalNER
from src.sentiment_intent import SentimentIntentAnalyzer


def example_1_extract_entities():
    """Example 1: Extract medical entities from a transcript"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Medical Entity Extraction")
    print("="*70)
    
    sample_text = """
    Patient: I had a car accident last month. My neck and back have been 
    hurting ever since. I went to the ER and they said it was whiplash.
    I've been taking painkillers and doing physiotherapy twice a week.
    The pain is getting better but I still have some stiffness.
    """
    
    ner = MedicalNER()
    entities = ner.extract_medical_entities(sample_text)
    
    print("\nInput Text:")
    print(sample_text)
    print("\nExtracted Entities:")
    print(json.dumps(entities, indent=2))


def example_2_sentiment_analysis():
    """Example 2: Analyze patient sentiment"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Sentiment Analysis")
    print("="*70)
    
    test_utterances = [
        "I'm really worried about whether this pain will ever go away.",
        "The treatment is working well, I feel much better now.",
        "I still have some discomfort, but it's manageable.",
        "Thank you doctor, that's very reassuring to hear.",
        "I'm afraid this might get worse over time."
    ]
    
    analyzer = SentimentIntentAnalyzer()
    
    for utterance in test_utterances:
        result = analyzer.analyze(utterance)
        print(f"\nText: \"{utterance}\"")
        print(f"→ Sentiment: {result['Sentiment']}")
        print(f"→ Intent: {result['Intent']}")


def example_3_compare_transcripts():
    """Example 3: Compare analysis of different transcripts"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Comparing Multiple Cases")
    print("="*70)
    
    cases = {
        "Acute Case": """
        Patient: I had a severe accident yesterday. The pain in my back is 
        unbearable. I can barely move. I'm very worried about permanent damage.
        """,
        
        "Recovering Case": """
        Patient: I'm doing much better after the physiotherapy. The pain has 
        reduced significantly. I can return to most of my normal activities now.
        """,
        
        "Follow-up Case": """
        Patient: I still have occasional discomfort, but nothing like before. 
        I'm mostly back to normal. Just want to make sure everything is healing properly.
        """
    }
    
    ner = MedicalNER()
    analyzer = SentimentIntentAnalyzer()
    
    for case_name, transcript in cases.items():
        print(f"\n--- {case_name} ---")
        
        # Extract entities
        entities = ner.extract_medical_entities(transcript)
        print(f"Symptoms: {', '.join(entities['Symptoms'][:3])}")
        print(f"Status: {entities['Current_Status']}")
        
        # Analyze sentiment
        sentiment_result = analyzer.analyze(transcript)
        print(f"Sentiment: {sentiment_result['Sentiment']}")
        print(f"Intent: {sentiment_result['Intent']}")


def example_4_batch_processing():
    """Example 4: Process multiple patient utterances"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Batch Processing")
    print("="*70)
    
    # Simulate multiple patient responses
    patient_responses = [
        "I've been experiencing severe headaches for the past week.",
        "The medication you prescribed has really helped with the pain.",
        "I'm a bit concerned about the side effects I've been having.",
        "How long will it take before I feel completely better?",
        "Thank you so much for your help, doctor."
    ]
    
    analyzer = SentimentIntentAnalyzer()
    
    results = []
    for i, response in enumerate(patient_responses, 1):
        analysis = analyzer.analyze(response)
        results.append({
            'id': i,
            'text': response,
            'sentiment': analysis['Sentiment'],
            'intent': analysis['Intent']
        })
    
    # Summary statistics
    sentiments = [r['sentiment'] for r in results]
    print(f"\nProcessed {len(results)} utterances")
    print(f"Anxious: {sentiments.count('Anxious')}")
    print(f"Neutral: {sentiments.count('Neutral')}")
    print(f"Reassured: {sentiments.count('Reassured')}")
    
    print("\nDetailed Results:")
    for result in results:
        print(f"{result['id']}. [{result['sentiment']}] {result['text'][:50]}...")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("MEDICAL NLP PIPELINE - USAGE EXAMPLES")
    print("="*70)
    
    example_1_extract_entities()
    example_2_sentiment_analysis()
    example_3_compare_transcripts()
    example_4_batch_processing()
    
    print("\n" + "="*70)
    print("Examples completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
