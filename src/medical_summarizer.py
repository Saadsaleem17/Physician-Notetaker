"""
Medical Transcript Summarization Module
Converts medical transcripts into structured reports.
"""

from typing import Dict, List
import re
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


class MedicalSummarizer:
    """Summarize medical transcripts into structured reports"""
    
    def __init__(self):
        """Initialize summarization model"""
        try:
            # Use a medical-focused or general summarization model
            self.summarizer = pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
        except Exception as e:
            print(f"Warning: Could not load summarization model: {e}")
            self.summarizer = None
    
    def extract_conversation_sections(self, text: str) -> Dict[str, str]:
        """
        Split conversation into logical sections
        """
        sections = {
            "chief_complaint": "",
            "history": "",
            "symptoms": "",
            "treatment_history": "",
            "current_status": "",
            "physical_exam": "",
            "assessment": "",
            "plan": ""
        }
        
        text_lower = text.lower()
        
        # Extract Chief Complaint
        complaint_patterns = [
            r'how are you feeling[^?]*\?[^.]*?patient:[^.]*?([^.]+\.)',
            r'what.*(?:brings|brought).*today[^?]*\?[^.]*?patient:[^.]*?([^.]+\.)',
        ]
        for pattern in complaint_patterns:
            match = re.search(pattern, text_lower, re.DOTALL)
            if match:
                sections["chief_complaint"] = match.group(1).strip()
                break
        
        # Extract History of Present Illness
        if 'car accident' in text_lower or 'accident' in text_lower:
            history_match = re.search(
                r'((?:car\s+)?accident[^.]+(?:\.[^.]+){0,3})',
                text_lower,
                re.DOTALL
            )
            if history_match:
                sections["history"] = history_match.group(1).strip()
        
        # Extract Physical Examination
        if 'physical examination' in text_lower or 'examination conducted' in text_lower:
            exam_match = re.search(
                r'(?:physical\s+examination[^.]*|examination\s+conducted)(.*?)(?:patient:|physician:|$)',
                text_lower,
                re.DOTALL
            )
            if exam_match:
                sections["physical_exam"] = exam_match.group(1).strip()
        
        return sections
    
    def create_structured_summary(self, text: str, entities: Dict) -> str:
        """
        Create a structured text summary of the medical encounter
        
        Args:
            text: Original transcript
            entities: Extracted medical entities from NER
            
        Returns:
            Structured summary text
        """
        summary_parts = []
        
        # Header
        summary_parts.append(f"MEDICAL CONSULTATION SUMMARY")
        summary_parts.append(f"Patient: {entities.get('Patient_Name', 'Unknown')}")
        summary_parts.append(f"Date of Incident: {entities.get('Date_of_Incident', 'Unknown')}")
        summary_parts.append("")
        
        # Chief Complaint
        summary_parts.append("CHIEF COMPLAINT:")
        symptoms = entities.get('Symptoms', [])
        if symptoms:
            summary_parts.append(f"  {', '.join(symptoms[:3])}")
        summary_parts.append("")
        
        # Diagnosis
        summary_parts.append("DIAGNOSIS:")
        summary_parts.append(f"  {entities.get('Diagnosis', 'Unknown')}")
        summary_parts.append("")
        
        # Treatment
        summary_parts.append("TREATMENT PROVIDED:")
        treatments = entities.get('Treatment', [])
        if treatments:
            for treatment in treatments:
                summary_parts.append(f"  - {treatment}")
        summary_parts.append("")
        
        # Current Status
        summary_parts.append("CURRENT STATUS:")
        summary_parts.append(f"  {entities.get('Current_Status', 'Unknown')}")
        summary_parts.append("")
        
        # Prognosis
        summary_parts.append("PROGNOSIS:")
        summary_parts.append(f"  {entities.get('Prognosis', 'Unknown')}")
        summary_parts.append("")
        
        return '\n'.join(summary_parts)
    
    def summarize_abstractive(self, text: str, max_length: int = 150) -> str:
        """
        Generate abstractive summary using transformer model
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        if not self.summarizer:
            return "Summarization model not available. Using extractive summary instead."
        
        try:
            # Clean text for better summarization
            clean_text = self._clean_transcript(text)
            
            # Generate summary
            if len(clean_text) > 1024:
                # Split into chunks if too long
                chunks = self._split_text(clean_text, 1024)
                summaries = []
                for chunk in chunks[:3]:  # Limit to first 3 chunks
                    result = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=30,
                        do_sample=False
                    )
                    summaries.append(result[0]['summary_text'])
                return ' '.join(summaries)
            else:
                result = self.summarizer(
                    clean_text,
                    max_length=max_length,
                    min_length=30,
                    do_sample=False
                )
                return result[0]['summary_text']
        except Exception as e:
            print(f"Error in summarization: {e}")
            return "Error generating summary."
    
    def _clean_transcript(self, text: str) -> str:
        """Clean transcript for summarization"""
        # Remove speaker labels
        text = re.sub(r'(?:Physician|Doctor|Patient):\s*', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks"""
        sentences = re.split(r'[.!?]\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length > max_length:
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def generate_json_summary(self, entities: Dict) -> Dict:
        """
        Generate JSON formatted summary
        
        Args:
            entities: Extracted medical entities
            
        Returns:
            JSON-formatted dictionary
        """
        return {
            "Patient_Name": entities.get("Patient_Name", "Unknown"),
            "Date_of_Incident": entities.get("Date_of_Incident", "Unknown"),
            "Symptoms": entities.get("Symptoms", []),
            "Diagnosis": entities.get("Diagnosis", "Unknown"),
            "Treatment": entities.get("Treatment", []),
            "Current_Status": entities.get("Current_Status", "Unknown"),
            "Prognosis": entities.get("Prognosis", "Unknown")
        }


if __name__ == "__main__":
    # Test the summarizer
    sample_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September.
    Patient: Yes, it was on September 1st. I had to go through ten sessions of 
    physiotherapy to help with the stiffness and discomfort.
    Physician: Everything looks good. I'd expect you to make a full recovery 
    within six months of the accident.
    """
    
    sample_entities = {
        "Patient_Name": "Janet Jones",
        "Date_of_Incident": "September 1st",
        "Symptoms": ["Neck pain", "Back pain", "Head impact"],
        "Diagnosis": "Whiplash injury",
        "Treatment": ["10 physiotherapy sessions", "Painkillers"],
        "Current_Status": "Occasional backache",
        "Prognosis": "Full recovery expected within six months"
    }
    
    summarizer = MedicalSummarizer()
    
    print("Structured Summary:")
    print(summarizer.create_structured_summary(sample_text, sample_entities))
    
    print("\n" + "="*70)
    print("\nJSON Summary:")
    import json
    print(json.dumps(summarizer.generate_json_summary(sample_entities), indent=2))
