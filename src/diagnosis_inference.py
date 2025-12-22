"""
Diagnosis Inference Layer
Infers diagnosis from transcript using controlled heuristics + weak semantics

This module sits AFTER NER extraction and uses:
- Linguistic triggers (appears to be, consistent with, likely)
- Symptom + body-part compatibility
- Speaker-aware prioritization (physician > patient)
- Negation awareness

Returns normalized diagnosis with confidence score and evidence.
"""

import re
from typing import Dict, List, Tuple, Optional


class DiagnosisInferenceLayer:
    """
    Infer diagnosis from medical transcript using rule-guided inference
    
    NOT rule-based extraction (too brittle)
    NOT pure LLM (hallucinates)
    HYBRID: Controlled heuristics with semantic soft-matching
    """
    
    def __init__(self):
        """Initialize diagnosis vocabulary and linguistic triggers"""
        
        # Curated diagnosis vocabulary with symptom/body-part compatibility
        self.diagnosis_vocab = {
            'whiplash': {
                'normalized': 'Whiplash Injury',
                'keywords': ['whiplash', 'whiplash injury'],
                'compatible_body_parts': ['neck', 'back', 'head'],
                'compatible_symptoms': ['neck pain', 'back pain', 'head impact', 'stiffness'],
            },
            'sprain': {
                'normalized': 'Sprain',
                'keywords': ['sprain', 'sprained'],
                'compatible_body_parts': ['wrist', 'ankle', 'knee', 'hand', 'foot'],
                'compatible_symptoms': ['pain', 'swelling', 'stiffness'],
            },
            'wrist_sprain': {
                'normalized': 'Wrist Sprain',
                'keywords': ['wrist sprain'],
                'compatible_body_parts': ['wrist', 'hand'],
                'compatible_symptoms': ['wrist pain', 'wrist swelling', 'wrist stiffness'],
            },
            'mechanical_back_pain': {
                'normalized': 'Mechanical Lower Back Pain',
                'keywords': ['mechanical back pain', 'mechanical lower back pain', 
                           'posture-related back pain', 'posture-related mechanical back pain'],
                'compatible_body_parts': ['back', 'lower back', 'spine'],
                'compatible_symptoms': ['back pain', 'lower back pain', 'stiffness'],
            },
            'soft_tissue_injury': {
                'normalized': 'Soft Tissue Injury',
                'keywords': ['soft tissue injury', 'resolving soft tissue injury'],
                'compatible_body_parts': ['wrist', 'hand', 'arm', 'leg', 'knee', 'ankle'],
                'compatible_symptoms': ['pain', 'swelling', 'stiffness'],
            },
            'strain': {
                'normalized': 'Strain',
                'keywords': ['strain', 'muscle strain', 'back strain', 'lower back strain'],
                'compatible_body_parts': ['back', 'neck', 'leg', 'arm'],
                'compatible_symptoms': ['pain', 'stiffness', 'ache'],
            },
            'rotator_cuff_shoulder_strain': {
                'normalized': 'Rotator Cuff-Related Shoulder Strain',
                'keywords': ['rotator cuff', 'rotator cuff related shoulder strain', 'shoulder strain'],
                'compatible_body_parts': ['shoulder', 'arm', 'upper arm'],
                'compatible_symptoms': ['shoulder pain', 'upper arm pain', 'weakness', 'pain on overhead movement'],
            },
            'concussion': {
                'normalized': 'Concussion',
                'keywords': ['concussion', 'minor concussion', 'head trauma'],
                'compatible_body_parts': ['head', 'brain'],
                'compatible_symptoms': ['headache', 'dizziness', 'head impact', 'confusion'],
            },
        }
        
        # Soft linguistic triggers for diagnosis mentions
        self.physician_triggers = [
            r'\b(?:this\s+)?(?:appears\s+to\s+be|is)\s+(?:a\s+|an\s+)?',
            r'\b(?:consistent\s+with|likely|suggests?)\s+(?:a\s+|an\s+)?',
            r'\b(?:diagnosed?\s+(?:with|as)|said\s+(?:it\s+)?was)\s+(?:a\s+|an\s+)?',
            r'\b(?:just\s+)?(?:a\s+|an\s+)',
            r'\bconfirmed\s+(?:a\s+|an\s+)?',
        ]
        
    def extract_physician_statements(self, text: str) -> List[str]:
        """
        Extract sentences where physician is speaking
        
        Args:
            text: Full transcript
            
        Returns:
            List of physician statements
        """
        statements = []
        
        # Split by speaker markers
        sections = re.split(r'\n\n(?=Physician:|Patient:)', text, flags=re.IGNORECASE)
        
        for section in sections:
            if section.strip().lower().startswith('physician:'):
                # Extract the statement
                statement = re.sub(r'^Physician:\s*', '', section, flags=re.IGNORECASE)
                statements.append(statement.strip())
        
        return statements
    
    def find_diagnosis_mentions(self, text: str, speaker: str = 'all') -> List[Tuple[str, str, float]]:
        """
        Find potential diagnosis mentions with confidence scoring
        
        Args:
            text: Input text
            speaker: 'all', 'physician', or 'patient'
            
        Returns:
            List of (diagnosis_key, matched_phrase, confidence) tuples
        """
        matches = []
        text_lower = text.lower()
        
        # Get speaker-specific text if needed
        if speaker == 'physician':
            physician_statements = self.extract_physician_statements(text)
            search_text = ' '.join(physician_statements).lower()
        else:
            search_text = text_lower
        
        # Search for each diagnosis in vocabulary
        for diag_key, diag_info in self.diagnosis_vocab.items():
            for keyword in diag_info['keywords']:
                # Check if keyword appears
                if keyword in search_text:
                    # Calculate confidence based on context
                    confidence = self._calculate_confidence(
                        keyword, 
                        search_text, 
                        text_lower,
                        speaker == 'physician'
                    )
                    
                    if confidence > 0.3:  # Minimum threshold
                        matches.append((diag_key, keyword, confidence))
        
        return matches
    
    def _calculate_confidence(self, keyword: str, search_text: str, 
                             full_text: str, is_physician: bool) -> float:
        """
        Calculate confidence score for diagnosis mention
        
        Factors:
        - Physician mention: +0.4
        - Linguistic trigger nearby: +0.3
        - Not negated: required (0 if negated)
        - Base confidence: 0.3
        
        Args:
            keyword: Diagnosis keyword
            search_text: Text to search in (may be physician-only)
            full_text: Full transcript
            is_physician: Whether this is from physician speech
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.3  # Base confidence
        
        # Check for negation first (disqualifies immediately)
        negation_pattern = rf'\b(?:no|not|without|rule\s+out)\s+(?:\w+\s+){{0,3}}{re.escape(keyword)}\b'
        if re.search(negation_pattern, full_text.lower()):
            return 0.0
        
        # Physician mention bonus
        if is_physician:
            confidence += 0.4
        
        # Check for linguistic triggers nearby
        for trigger in self.physician_triggers:
            pattern = rf'{trigger}(?:\w+\s+){{0,3}}{re.escape(keyword)}\b'
            if re.search(pattern, search_text):
                confidence += 0.3
                break
        
        return min(confidence, 1.0)
    
    def check_symptom_compatibility(self, diagnosis_key: str, symptoms: List[str], 
                                   body_parts: List[str]) -> float:
        """
        Check if extracted symptoms/body parts are compatible with diagnosis
        
        Args:
            diagnosis_key: Key in diagnosis_vocab
            symptoms: List of extracted symptoms
            body_parts: List of mentioned body parts
            
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        if diagnosis_key not in self.diagnosis_vocab:
            return 0.0
        
        diag_info = self.diagnosis_vocab[diagnosis_key]
        score = 0.0
        
        # Check body part compatibility
        symptoms_lower = [s.lower() for s in symptoms]
        body_parts_lower = [b.lower() for b in body_parts]
        
        compatible_body_parts = diag_info['compatible_body_parts']
        compatible_symptoms = diag_info['compatible_symptoms']
        
        # Body part match
        body_matches = sum(1 for bp in body_parts_lower 
                          if any(cbp in bp for cbp in compatible_body_parts))
        if body_matches > 0:
            score += 0.3
        
        # Symptom match
        symptom_matches = sum(1 for sym in symptoms_lower 
                             if any(cs in sym for cs in compatible_symptoms))
        if symptom_matches > 0:
            score += 0.4
        
        # Exact match bonus
        if symptom_matches >= 2 or body_matches >= 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def infer_diagnosis(self, transcript: str, extracted_symptoms: List[str], 
                       extracted_body_parts: List[str] = None) -> Dict:
        """
        Main inference method: infer diagnosis from all available signals
        
        Args:
            transcript: Full medical transcript
            extracted_symptoms: Symptoms from NER
            extracted_body_parts: Body parts mentioned (optional, extracted if None)
            
        Returns:
            {
                'diagnosis': 'Normalized Diagnosis Name' or '',
                'confidence': float (0.0 to 1.0),
                'evidence_sentences': List[str],
                'reasoning': str
            }
        """
        # Extract body parts if not provided
        if extracted_body_parts is None:
            extracted_body_parts = self._extract_body_parts_simple(transcript)
        
        # Find diagnosis mentions from physician statements (prioritized)
        physician_mentions = self.find_diagnosis_mentions(transcript, speaker='physician')
        
        # Find all mentions as backup
        all_mentions = self.find_diagnosis_mentions(transcript, speaker='all')
        
        # Score and rank candidates
        candidates = []
        
        # Prioritize physician mentions
        for diag_key, keyword, base_confidence in physician_mentions:
            # Check compatibility with extracted symptoms/body parts
            compat_score = self.check_symptom_compatibility(
                diag_key, extracted_symptoms, extracted_body_parts
            )
            
            # Combined confidence: 70% from mention, 30% from compatibility
            final_confidence = (base_confidence * 0.7) + (compat_score * 0.3)
            
            # Extract evidence sentence
            evidence = self._extract_evidence_sentence(transcript, keyword)
            
            candidates.append({
                'diagnosis_key': diag_key,
                'normalized': self.diagnosis_vocab[diag_key]['normalized'],
                'confidence': final_confidence,
                'evidence': evidence,
                'keyword': keyword,
                'source': 'physician'
            })
        
        # If no physician mentions, check all mentions
        if not candidates:
            for diag_key, keyword, base_confidence in all_mentions:
                compat_score = self.check_symptom_compatibility(
                    diag_key, extracted_symptoms, extracted_body_parts
                )
                
                final_confidence = (base_confidence * 0.6) + (compat_score * 0.4)
                evidence = self._extract_evidence_sentence(transcript, keyword)
                
                candidates.append({
                    'diagnosis_key': diag_key,
                    'normalized': self.diagnosis_vocab[diag_key]['normalized'],
                    'confidence': final_confidence,
                    'evidence': evidence,
                    'keyword': keyword,
                    'source': 'general'
                })
        
        # Select best candidate
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            
            return {
                'diagnosis': best['normalized'],
                'confidence': round(best['confidence'], 2),
                'evidence_sentences': [best['evidence']],
                'reasoning': f"Detected '{best['keyword']}' from {best['source']} speech with {round(best['confidence']*100)}% confidence"
            }
        
        # No diagnosis found
        return {
            'diagnosis': '',
            'confidence': 0.0,
            'evidence_sentences': [],
            'reasoning': 'No diagnosis mentioned or insufficient evidence'
        }
    
    def _extract_body_parts_simple(self, text: str) -> List[str]:
        """Simple body part extraction as fallback"""
        body_parts = []
        text_lower = text.lower()
        
        common_parts = ['wrist', 'hand', 'neck', 'back', 'head', 'chest', 'shoulder', 
                       'knee', 'ankle', 'elbow', 'hip', 'foot', 'leg', 'arm', 'spine']
        
        for part in common_parts:
            if part in text_lower:
                body_parts.append(part)
        
        return body_parts
    
    def _extract_evidence_sentence(self, text: str, keyword: str) -> str:
        """Extract the sentence containing the diagnosis keyword"""
        # Find the sentence containing the keyword
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence.strip()
        
        return ""


if __name__ == "__main__":
    # Test the inference layer
    inference = DiagnosisInferenceLayer()
    
    # Test case 1: Mechanical back pain
    transcript1 = """
    Physician: What brings you in today?
    Patient: I have lower back pain.
    Physician: Your spinal movement is normal with no neurological signs. 
    This appears to be posture-related mechanical back pain.
    """
    
    result1 = inference.infer_diagnosis(
        transcript1, 
        extracted_symptoms=['Back Pain'],
        extracted_body_parts=['back']
    )
    
    print("Test 1: Mechanical Back Pain")
    print(f"Diagnosis: {result1['diagnosis']}")
    print(f"Confidence: {result1['confidence']}")
    print(f"Evidence: {result1['evidence_sentences']}")
    print(f"Reasoning: {result1['reasoning']}")
    print()
    
    # Test case 2: Wrist sprain
    transcript2 = """
    Physician: Did you seek medical care?
    Patient: Yes, they took an X-ray and said there was no fracture, just a sprain.
    """
    
    result2 = inference.infer_diagnosis(
        transcript2,
        extracted_symptoms=['Wrist Pain', 'Wrist Swelling'],
        extracted_body_parts=['wrist']
    )
    
    print("Test 2: Wrist Sprain")
    print(f"Diagnosis: {result2['diagnosis']}")
    print(f"Confidence: {result2['confidence']}")
    print(f"Evidence: {result2['evidence_sentences']}")
    print(f"Reasoning: {result2['reasoning']}")
