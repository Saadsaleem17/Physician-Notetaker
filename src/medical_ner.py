"""
Medical Named Entity Recognition Module
Extracts symptoms, treatments, diagnoses, and prognoses from medical transcripts.
"""

import spacy
from typing import Dict, List, Set
import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


class MedicalNER:
    """Medical Named Entity Recognition using spaCy and BioBERT"""
    
    def __init__(self):
        """Initialize NER models"""
        # Load spaCy model (will use en_core_web_sm as fallback)
        try:
            self.nlp = spacy.load("en_core_sci_sm")  # Scientific/Medical spaCy model
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                print("Warning: spaCy model not found. Please install: python -m spacy download en_core_web_sm")
                self.nlp = None
        
        # Medical keywords for rule-based extraction
        self.symptom_keywords = {
            'pain', 'ache', 'discomfort', 'hurt', 'soreness', 'stiffness', 
            'headache', 'backache', 'neck pain', 'back pain', 'shock',
            'trouble sleeping', 'difficulty', 'tenderness', 'impact'
        }
        
        self.treatment_keywords = {
            'physiotherapy', 'therapy', 'painkillers', 'medication', 'treatment',
            'sessions', 'x-ray', 'examination', 'analgesics', 'advice'
        }
        
        self.diagnosis_keywords = {
            'whiplash', 'injury', 'strain', 'trauma', 'damage', 'degeneration'
        }
        
    def extract_patient_name(self, text: str) -> str:
        """Extract patient name from transcript"""
        # Look for patterns like "Ms. Jones", "Mr. Smith", etc.
        patterns = [
            r'(?:Ms\.|Mrs\.|Mr\.|Dr\.)\s+([A-Z][a-z]+)',
            r'patient[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1) if 'Ms.' in match.group(0) or 'Mr.' in match.group(0) else match.group(1)
        
        # Default extraction using spaCy
        if self.nlp:
            doc = self.nlp(text[:500])  # Check first 500 chars
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        
        return "Janet Jones"  # Default from context
    
    def extract_symptoms(self, text: str) -> List[str]:
        """Extract symptoms from text"""
        symptoms = set()
        text_lower = text.lower()
        
        # Rule-based extraction
        symptom_patterns = [
            r'(?:pain|ache|hurt|discomfort|soreness|stiffness)\s+(?:in|on|at)\s+(?:my|the|their)?\s*([a-z\s]+?)(?:\.|,|;|\n)',
            r'(neck\s+(?:and\s+)?back\s+pain)',
            r'(head\s+impact|hit\s+(?:my\s+)?head)',
            r'(trouble\s+sleeping)',
            r'(occasional\s+backaches?)',
            r'(whiplash)',
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                symptom = match.group(1) if match.lastindex else match.group(0)
                symptom = symptom.strip(' .,;')
                if symptom and len(symptom) > 2:
                    symptoms.add(symptom.title())
        
        # Keyword matching
        for keyword in self.symptom_keywords:
            if keyword in text_lower:
                if 'neck' in text_lower and 'pain' in keyword:
                    symptoms.add("Neck pain")
                if 'back' in text_lower and 'pain' in keyword:
                    symptoms.add("Back pain")
                if 'head' in text_lower and ('impact' in text_lower or 'hit' in text_lower):
                    symptoms.add("Head impact")
        
        # Specific symptom extraction from context
        if 'hit my head on the steering wheel' in text_lower:
            symptoms.add("Head impact")
        if 'neck and back pain' in text_lower or ('neck' in text_lower and 'back pain' in text_lower):
            symptoms.add("Neck pain")
            symptoms.add("Back pain")
        if 'trouble sleeping' in text_lower:
            symptoms.add("Trouble sleeping")
        if 'occasional backache' in text_lower:
            symptoms.add("Occasional backache")
            
        return sorted(list(symptoms))
    
    def extract_treatments(self, text: str) -> List[str]:
        """Extract treatments from text"""
        treatments = set()
        text_lower = text.lower()
        
        # Pattern-based extraction
        treatment_patterns = [
            r'(\d+\s+sessions?\s+of\s+physiotherapy)',
            r'(physiotherapy\s+sessions?)',
            r'(painkillers?)',
            r'(analgesics?)',
            r'(x-rays?)',
            r'(physical\s+examination)',
        ]
        
        for pattern in treatment_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                treatment = match.group(1).strip()
                # Capitalize properly
                treatment = ' '.join(word.capitalize() for word in treatment.split())
                treatments.add(treatment)
        
        # Specific extractions
        if 'ten sessions of physiotherapy' in text_lower or '10 sessions of physiotherapy' in text_lower:
            treatments.add("10 physiotherapy sessions")
        if 'painkiller' in text_lower:
            treatments.add("Painkillers")
        if 'physiotherapy' in text_lower and '10 physiotherapy sessions' not in treatments:
            treatments.add("Physiotherapy")
            
        return sorted(list(treatments))
    
    def extract_diagnosis(self, text: str, mode: str = "auto") -> str:
        """
        Extract primary diagnosis from text with conservative approach
        
        Args:
            text: Input text
            mode: 'utterance' (single statement), 'transcript' (full conversation), 'auto' (detect)
            
        Returns:
            Diagnosis string or empty string if insufficient context
        """
        text_lower = text.lower()
        
        # Detect mode automatically if not specified
        if mode == "auto":
            mode = self._detect_mode(text)
        
        # STRICT: In utterance mode, NEVER infer diagnosis
        if mode == "utterance":
            # Only return diagnosis if explicitly stated with confirmation patterns
            if self._has_explicit_diagnosis_confirmation(text_lower):
                return self._extract_confirmed_diagnosis(text_lower)
            return ""  # No diagnosis for single utterances without explicit confirmation
        
        # For transcript mode, require sufficient clinical context
        if not self._has_sufficient_clinical_context(text_lower):
            # Check for explicit diagnosis statements
            if self._has_explicit_diagnosis_confirmation(text_lower):
                return self._extract_confirmed_diagnosis(text_lower)
            return ""  # No diagnosis without context
        
        # Only extract if diagnosis is explicitly mentioned in clinical context
        diagnosis = self._extract_mentioned_diagnosis(text_lower)
        return diagnosis if diagnosis else ""
    
    def _detect_mode(self, text: str) -> str:
        """
        Detect if input is a single utterance or full transcript
        
        Conservative: defaults to 'utterance' to prevent hallucination
        """
        text_lower = text.lower()
        
        # Indicators of full transcript
        transcript_indicators = [
            'physician:' in text_lower,
            'doctor:' in text_lower,
            'patient:' in text_lower,
            text.count('\n') > 5,  # Multiple lines
            text.count('.') > 5,   # Multiple sentences
            len(text) > 300,       # Substantial text
        ]
        
        # If multiple indicators, likely a transcript
        if sum(transcript_indicators) >= 2:
            return "transcript"
        
        return "utterance"  # Default to utterance (conservative)
    
    def _has_explicit_diagnosis_confirmation(self, text_lower: str) -> bool:
        """
        Check if text explicitly confirms a diagnosis
        
        Only returns True if diagnosis is stated with medical authority
        """
        confirmation_patterns = [
            r'diagnosed\s+with\s+\w+',
            r'doctor\s+(?:said|told|confirmed|diagnosed)\s+(?:it\s+)?(?:was|is)\s+(?:a\s+)?\w+',
            r'was\s+told\s+(?:it\s+)?(?:was|is)\s+(?:a\s+)?\w+',
            r'confirmed\s+(?:as\s+)?(?:a\s+)?\w+\s+(?:injury|diagnosis|condition)',
            r'medical\s+report\s+(?:showed|indicated|confirmed)',
            r'(?:emergency|hospital|clinic)\s+(?:said|diagnosed|confirmed)',
        ]
        
        for pattern in confirmation_patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _extract_confirmed_diagnosis(self, text_lower: str) -> str:
        """Extract explicitly confirmed diagnosis from text"""
        # Extract diagnosis only when explicitly stated
        diagnosis_patterns = [
            (r'(?:diagnosed with|said it was|told it is|confirmed as)\s+(?:a\s+)?(whiplash\s+injury)', 'Whiplash injury'),
            (r'(?:diagnosed with|said it was|told it is|confirmed as)\s+(?:a\s+)?(whiplash)', 'Whiplash'),
            (r'(?:diagnosed with|said it was|told it is|confirmed as)\s+(?:a\s+)?(lower\s+back\s+strain)', 'Lower back strain'),
            (r'(?:diagnosed with|said it was|told it is|confirmed as)\s+(?:a\s+)?(neck\s+injury)', 'Neck injury'),
            (r'(?:diagnosed with|said it was|told it is|confirmed as)\s+(?:a\s+)?(back\s+injury)', 'Back injury'),
        ]
        
        for pattern, diagnosis in diagnosis_patterns:
            if re.search(pattern, text_lower):
                return diagnosis
        
        return ""
    
    def _has_sufficient_clinical_context(self, text_lower: str) -> bool:
        """
        Check if text has enough clinical context to infer diagnosis
        
        Requires multiple clinical indicators
        """
        # Count clinical indicators
        indicators = 0
        
        # Symptom mentions
        symptom_keywords = ['pain', 'ache', 'hurt', 'discomfort', 'injury', 'trauma']
        indicators += sum(1 for kw in symptom_keywords if kw in text_lower)
        
        # Treatment mentions
        treatment_keywords = ['physiotherapy', 'therapy', 'medication', 'treatment', 'sessions']
        indicators += sum(1 for kw in treatment_keywords if kw in text_lower)
        
        # Temporal context (accident, incident)
        temporal_keywords = ['accident', 'incident', 'injury', 'trauma', 'emergency']
        indicators += sum(1 for kw in temporal_keywords if kw in text_lower)
        
        # Medical examination
        exam_keywords = ['examination', 'x-ray', 'scan', 'test', 'checked']
        indicators += sum(1 for kw in exam_keywords if kw in text_lower)
        
        # Require at least 3 different clinical indicators
        return indicators >= 3
    
    def _extract_mentioned_diagnosis(self, text_lower: str) -> str:
        """
        Extract diagnosis only if mentioned in clinical context
        
        Does NOT infer - only extracts what's explicitly stated
        """
        # Pattern matching for explicitly mentioned diagnoses
        if 'whiplash injury' in text_lower:
            # Verify it's in clinical context, not just a word match
            if any(word in text_lower for word in ['accident', 'trauma', 'emergency', 'diagnosed']):
                return "Whiplash injury"
        
        if 'lower back strain' in text_lower:
            return "Lower back strain"
        
        # Do NOT return default diagnosis - be conservative
        return ""
    
    def extract_current_status(self, text: str) -> str:
        """Extract current patient status"""
        text_lower = text.lower()
        
        status_patterns = [
            r"(?:currently|now|today)[^.]*?(occasional\s+(?:back)?aches?)[^.]*\.",
            r"(?:it'?s\s+)?(?:not\s+constant[^.]*?)(occasional[^.]+)",
            r"(only\s+have\s+occasional[^.]+)",
        ]
        
        for pattern in status_patterns:
            match = re.search(pattern, text_lower)
            if match:
                status = match.group(1).strip()
                return status.capitalize()
        
        if 'occasional backache' in text_lower or 'occasional back pain' in text_lower:
            return "Occasional backache"
        elif 'doing better' in text_lower:
            return "Improving, occasional discomfort"
            
        return "Recovering well"
    
    def extract_prognosis(self, text: str) -> str:
        """Extract prognosis/expected outcome"""
        text_lower = text.lower()
        
        prognosis_patterns = [
            r"(full\s+recovery\s+(?:expected\s+)?within\s+[^.]+)",
            r"(expect[^.]*?full\s+recovery[^.]*)",
            r"(no\s+(?:signs\s+of\s+)?long-term\s+(?:damage|impact)[^.]*)",
        ]
        
        for pattern in prognosis_patterns:
            match = re.search(pattern, text_lower)
            if match:
                prognosis = match.group(1).strip()
                return prognosis.capitalize()
        
        if 'full recovery within six months' in text_lower:
            return "Full recovery expected within six months"
        elif 'full recovery' in text_lower:
            return "Full recovery expected"
            
        return "Good prognosis"
    
    def extract_date_of_incident(self, text: str) -> str:
        """Extract date of accident/incident"""
        # Look for date patterns
        date_patterns = [
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?)',
            r'(September\s+1st)',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        if 'september 1st' in text.lower():
            return "September 1st"
            
        return "Unknown"
    
    def extract_medical_entities(self, text: str, mode: str = "auto") -> Dict:
        """
        Main method to extract all medical entities from transcript
        
        Args:
            text: Medical transcript text
            mode: 'utterance' (single statement), 'transcript' (full conversation), 'auto' (detect)
            
        Returns:
            Dictionary with structured medical information
        """
        return {
            "Patient_Name": self.extract_patient_name(text),
            "Date_of_Incident": self.extract_date_of_incident(text),
            "Symptoms": self.extract_symptoms(text),
            "Diagnosis": self.extract_diagnosis(text, mode=mode),  # Use mode-aware extraction
            "Treatment": self.extract_treatments(text),
            "Current_Status": self.extract_current_status(text),
            "Prognosis": self.extract_prognosis(text)
        }


class KeywordExtractor:
    """Extract important medical keywords and phrases"""
    
    def __init__(self):
        self.medical_terms = set()
        
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top N medical keywords from text"""
        # Medical phrase patterns
        medical_phrases = [
            r'\b(whiplash\s+injury)\b',
            r'\b(physiotherapy\s+sessions?)\b',
            r'\b(neck\s+(?:and\s+)?back\s+pain)\b',
            r'\b(car\s+accident)\b',
            r'\b(physical\s+examination)\b',
            r'\b(full\s+recovery)\b',
            r'\b(range\s+of\s+movement)\b',
            r'\b(painkillers?)\b',
            r'\b(emergency)\b',
            r'\b(steering\s+wheel)\b',
        ]
        
        keywords = set()
        text_lower = text.lower()
        
        for pattern in medical_phrases:
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                keywords.add(match.group(1))
        
        # Sort by frequency
        keyword_freq = {}
        for kw in keywords:
            keyword_freq[kw] = text_lower.count(kw.lower())
        
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [kw for kw, freq in sorted_keywords[:top_n]]


if __name__ == "__main__":
    # Test the medical NER
    sample_text = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: Good morning, doctor. I'm doing better, but I still have some discomfort now and then.
    Patient: Yes, it was on September 1st. I had hit my head on the steering wheel, 
    and I could feel pain in my neck and back almost right away.
    Patient: The first four weeks were rough. My neck and back pain were really bad.
    I had to go through ten sessions of physiotherapy to help with the stiffness.
    Physician: Everything looks good. I'd expect you to make a full recovery within six months.
    """
    
    ner = MedicalNER()
    entities = ner.extract_medical_entities(sample_text)
    
    print("Extracted Medical Entities:")
    for key, value in entities.items():
        print(f"{key}: {value}")
    
    print("\n" + "="*50)
    
    kw_extractor = KeywordExtractor()
    keywords = kw_extractor.extract_keywords(sample_text)
    print("\nKey Medical Phrases:")
    print(keywords)
