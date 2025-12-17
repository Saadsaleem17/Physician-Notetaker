"""
Medical Transcription NLP Pipeline
Main application integrating all modules for medical transcript analysis.
"""

import json
import sys
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.medical_ner import MedicalNER, KeywordExtractor
from src.medical_summarizer import MedicalSummarizer
from src.sentiment_intent import SentimentIntentAnalyzer
from src.soap_generator import SOAPNoteGenerator


class MedicalTranscriptionPipeline:
    """
    Complete NLP pipeline for medical transcription analysis
    
    Features:
    - Named Entity Recognition (NER)
    - Medical Summarization
    - Sentiment & Intent Analysis
    - SOAP Note Generation
    """
    
    def __init__(self):
        """Initialize all components"""
        print("Initializing Medical Transcription Pipeline...")
        
        self.ner = MedicalNER()
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = MedicalSummarizer()
        self.sentiment_analyzer = SentimentIntentAnalyzer()
        self.soap_generator = SOAPNoteGenerator()
        
        print("✓ Pipeline initialized successfully!\n")
    
    def process_transcript(self, transcript: str, verbose: bool = True) -> Dict:
        """
        Process a medical transcript through the complete pipeline
        
        Args:
            transcript: Medical conversation transcript
            verbose: Print progress messages
            
        Returns:
            Dictionary with all analysis results
        """
        results = {}
        
        # 1. Named Entity Recognition
        if verbose:
            print("📍 Step 1: Extracting Medical Entities (NER)...")
        entities = self.ner.extract_medical_entities(transcript)
        results['entities'] = entities
        if verbose:
            print(f"   Extracted: {len(entities.get('Symptoms', []))} symptoms, "
                  f"{len(entities.get('Treatment', []))} treatments")
        
        # 2. Keyword Extraction
        if verbose:
            print("📍 Step 2: Extracting Medical Keywords...")
        keywords = self.keyword_extractor.extract_keywords(transcript)
        results['keywords'] = keywords
        if verbose:
            print(f"   Found {len(keywords)} key medical phrases")
        
        # 3. Medical Summarization
        if verbose:
            print("📍 Step 3: Generating Medical Summary...")
        structured_summary = self.summarizer.create_structured_summary(transcript, entities)
        json_summary = self.summarizer.generate_json_summary(entities)
        results['structured_summary'] = structured_summary
        results['json_summary'] = json_summary
        if verbose:
            print("   ✓ Summary generated")
        
        # 4. Sentiment & Intent Analysis
        if verbose:
            print("📍 Step 4: Analyzing Sentiment & Intent...")
        conversation_analysis = self.sentiment_analyzer.analyze_conversation(transcript)
        results['sentiment_intent'] = conversation_analysis
        if verbose:
            print(f"   Analyzed {len(conversation_analysis)} patient utterances")
        
        # 5. SOAP Note Generation
        if verbose:
            print("📍 Step 5: Generating SOAP Note...")
        soap_note = self.soap_generator.generate_soap_note(transcript, entities)
        soap_text = self.soap_generator.format_soap_note_text(
            soap_note, 
            entities.get('Patient_Name')
        )
        results['soap_note'] = soap_note
        results['soap_note_text'] = soap_text
        if verbose:
            print("   ✓ SOAP note generated")
        
        if verbose:
            print("\n✅ Processing complete!\n")
        
        return results
    
    def analyze_single_utterance(self, text: str) -> Dict:
        """
        Analyze a single patient utterance for sentiment and intent
        
        Args:
            text: Patient dialogue text
            
        Returns:
            Dictionary with sentiment and intent
        """
        return self.sentiment_analyzer.analyze(text)
    
    def save_results(self, results: Dict, output_path: str):
        """
        Save analysis results to JSON file
        
        Args:
            results: Analysis results dictionary
            output_path: Path to output JSON file
        """
        # Convert non-serializable objects
        save_data = {
            'entities': results['entities'],
            'keywords': results['keywords'],
            'json_summary': results['json_summary'],
            'sentiment_intent': results['sentiment_intent'],
            'soap_note': results['soap_note']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Results saved to: {output_path}")


def main():
    """Main function demonstrating the pipeline"""
    
    print("=" * 70)
    print("Medical Transcription NLP Pipeline")
    print("=" * 70)
    print()
    
    # Load sample transcript
    transcript_path = Path(__file__).parent / 'data' / 'sample_transcript.json'
    
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            transcript = data['transcript_full']
    except FileNotFoundError:
        print(f"Error: Could not find {transcript_path}")
        print("Using default sample transcript...")
        transcript = """
        Physician: Good morning, Ms. Jones. How are you feeling today?
        Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
        Physician: I understand you were in a car accident last September.
        Patient: Yes, it was on September 1st. I had to go through ten sessions of 
        physiotherapy to help with the stiffness and discomfort.
        Physician: Everything looks good. I'd expect you to make a full recovery 
        within six months of the accident.
        """
    
    # Initialize pipeline
    pipeline = MedicalTranscriptionPipeline()
    
    # Process transcript
    results = pipeline.process_transcript(transcript, verbose=True)
    
    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print("\n1. EXTRACTED ENTITIES:")
    print(json.dumps(results['json_summary'], indent=2))
    
    print("\n2. KEY MEDICAL PHRASES:")
    for i, keyword in enumerate(results['keywords'], 1):
        print(f"   {i}. {keyword}")
    
    print("\n3. SENTIMENT & INTENT ANALYSIS:")
    for i, analysis in enumerate(results['sentiment_intent'], 1):
        print(f"\n   Utterance {i}:")
        print(f"   Text: {analysis['Text']}")
        print(f"   Sentiment: {analysis['Sentiment']}")
        print(f"   Intent: {analysis['Intent']}")
    
    print("\n4. SOAP NOTE:")
    print(results['soap_note_text'])
    
    # Save results
    output_path = Path(__file__).parent / 'output' / 'analysis_results.json'
    output_path.parent.mkdir(exist_ok=True)
    pipeline.save_results(results, str(output_path))
    
    # Example: Analyze single utterance
    print("\n" + "=" * 70)
    print("SINGLE UTTERANCE ANALYSIS EXAMPLE")
    print("=" * 70)
    
    sample_utterance = "I'm a bit worried about my back pain, but I hope it gets better soon."
    print(f"\nText: \"{sample_utterance}\"")
    
    analysis = pipeline.analyze_single_utterance(sample_utterance)
    print(f"Sentiment: {analysis['Sentiment']}")
    print(f"Intent: {analysis['Intent']}")
    
    print("\n" + "=" * 70)
    print("Pipeline execution completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
