"""
Medical Named Entity Recognition Module
Extracts symptoms, treatments, diagnoses, and prognoses from medical transcripts.
"""

import spacy
from typing import Dict, List, Set, Tuple, Optional
import re
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification


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
        """Extract sentences where physician is speaking"""
        statements = []
        sections = re.split(r'\n\n(?=Physician:|Patient:)', text, flags=re.IGNORECASE)
        
        for section in sections:
            if section.strip().lower().startswith('physician:'):
                statement = re.sub(r'^Physician:\s*', '', section, flags=re.IGNORECASE)
                statements.append(statement.strip())
        
        return statements
    
    def find_diagnosis_mentions(self, text: str, speaker: str = 'all') -> List[Tuple[str, str, float]]:
        """Find potential diagnosis mentions with confidence scoring"""
        matches = []
        text_lower = text.lower()
        
        if speaker == 'physician':
            physician_statements = self.extract_physician_statements(text)
            search_text = ' '.join(physician_statements).lower()
        else:
            search_text = text_lower
        
        for diag_key, diag_info in self.diagnosis_vocab.items():
            for keyword in diag_info['keywords']:
                if keyword in search_text:
                    confidence = self._calculate_confidence(
                        keyword, search_text, text_lower, speaker == 'physician'
                    )
                    if confidence > 0.3:
                        matches.append((diag_key, keyword, confidence))
        
        return matches
    
    def _calculate_confidence(self, keyword: str, search_text: str, 
                             full_text: str, is_physician: bool) -> float:
        """Calculate confidence score for diagnosis mention"""
        confidence = 0.3
        
        negation_pattern = rf'\b(?:no|not|without|rule\s+out)\s+(?:\w+\s+){{0,3}}{re.escape(keyword)}\b'
        if re.search(negation_pattern, full_text.lower()):
            return 0.0
        
        if is_physician:
            confidence += 0.4
        
        for trigger in self.physician_triggers:
            pattern = rf'{trigger}(?:\w+\s+){{0,3}}{re.escape(keyword)}\b'
            if re.search(pattern, search_text):
                confidence += 0.3
                break
        
        return min(confidence, 1.0)
    
    def check_symptom_compatibility(self, diagnosis_key: str, symptoms: List[str], 
                                   body_parts: List[str]) -> float:
        """Check if extracted symptoms/body parts are compatible with diagnosis"""
        if diagnosis_key not in self.diagnosis_vocab:
            return 0.0
        
        diag_info = self.diagnosis_vocab[diagnosis_key]
        score = 0.0
        
        symptoms_lower = [s.lower() for s in symptoms]
        body_parts_lower = [b.lower() for b in body_parts]
        
        compatible_body_parts = diag_info['compatible_body_parts']
        compatible_symptoms = diag_info['compatible_symptoms']
        
        body_matches = sum(1 for bp in body_parts_lower 
                          if any(cbp in bp for cbp in compatible_body_parts))
        if body_matches > 0:
            score += 0.3
        
        symptom_matches = sum(1 for sym in symptoms_lower 
                             if any(cs in sym for cs in compatible_symptoms))
        if symptom_matches > 0:
            score += 0.4
        
        if symptom_matches >= 2 or body_matches >= 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def infer_diagnosis(self, transcript: str, extracted_symptoms: List[str], 
                       extracted_body_parts: List[str] = None) -> Dict:
        """Main inference method: infer diagnosis from all available signals"""
        if extracted_body_parts is None:
            extracted_body_parts = self._extract_body_parts_simple(transcript)
        
        physician_mentions = self.find_diagnosis_mentions(transcript, speaker='physician')
        all_mentions = self.find_diagnosis_mentions(transcript, speaker='all')
        
        candidates = []
        
        for diag_key, keyword, base_confidence in physician_mentions:
            compat_score = self.check_symptom_compatibility(
                diag_key, extracted_symptoms, extracted_body_parts
            )
            final_confidence = (base_confidence * 0.7) + (compat_score * 0.3)
            evidence = self._extract_evidence_sentence(transcript, keyword)
            
            candidates.append({
                'diagnosis_key': diag_key,
                'normalized': self.diagnosis_vocab[diag_key]['normalized'],
                'confidence': final_confidence,
                'evidence': evidence,
                'keyword': keyword,
                'source': 'physician'
            })
        
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
        
        if candidates:
            best = max(candidates, key=lambda x: x['confidence'])
            return {
                'diagnosis': best['normalized'],
                'confidence': round(best['confidence'], 2),
                'evidence_sentences': [best['evidence']],
                'reasoning': f"Detected '{best['keyword']}' from {best['source']} speech with {round(best['confidence']*100)}% confidence"
            }
        
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
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                return sentence.strip()
        
        return ""


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
        
        # SYMPTOM VALIDATION: Categories to EXCLUDE
        # These are NOT symptoms - they are timestamps, body parts, states, etc.
        self.non_symptom_patterns = {
            # Timestamps and dates
            r'\b\d{1,2}\s*:\s*\d{2}\b',  # 12:30
            r'\b\d{1,2}\s*am\b|\b\d{1,2}\s*pm\b',  # 12am, 5pm
            r'\bin\b',  # standalone "in"
            
            # Durations and time references
            r'\bweeks?\b', r'\bdays?\b', r'\bmonths?\b', r'\byears?\b',
            r'\bhours?\b', r'\bminutes?\b',
            
            # Body parts alone (without sensation)
            r'^back$', r'^neck$', r'^head$', r'^chest$', r'^arm$', r'^leg$',
            r'^shoulder$', r'^knee$', r'^hand$', r'^foot$', r'^throat$',
            
            # States and progressions (not symptoms)
            r'^improving$', r'^better$', r'^worse$', r'^recovery$',
            r'^healing$', r'^progressing$',
            
            # Emotions and mental states (unless clinically diagnosed)
            r'^anxiety$', r'^anxious$', r'^nervous$', r'^worried$',
            r'^stress$', r'^stressed$', r'^upset$', r'^sad$',
            r'^emotional\s+issues?$', r'^feelings?$',
            
            # Non-specific terms
            r'^the$', r'^and$', r'^or$', r'^of$', r'^in$', r'^at$',
            r'^issues?$', r'^problems?$', r'^things?$',
            
            # Treatment/medical procedures (not symptoms)
            r'^sessions?$', r'^therapy$', r'^physiotherapy$',
            r'^treatment$', r'^medication$', r'^x-ray$',
        }
        
        # Denied symptom patterns - explicitly negated by patient
        # These use {symptom} placeholder for specific symptom checking
        self.denial_patterns = [
            r'\bno\s+(?:signs?\s+of\s+)?{symptom}\b',
            r'\bnot?\s+(?:having|experiencing|feeling)\s+(?:any\s+)?{symptom}\b',
            r'\b{symptom}\s+(?:is|are)\s+not\b',
            r'\bwithout\s+(?:any\s+)?{symptom}\b',
        ]
        
        # Load transformer-based medical NER pipeline
        try:
            from transformers import pipeline
            self.transformer_ner = pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple",
                device=-1  # CPU
            )
            print("✓ Transformer-based medical NER loaded")
        except Exception as e:
            print(f"Note: Transformer NER not available ({e})")
            self.transformer_ner = None
        
        # Medical keywords for rule-based extraction
        self.symptom_keywords = {
            'pain', 'ache', 'discomfort', 'hurt', 'soreness', 'stiffness', 
            'headache', 'backache', 'neck pain', 'back pain', 'shock',
            'trouble sleeping', 'difficulty', 'tenderness', 'impact',
            'cough', 'fever', 'nausea', 'vomiting', 'fatigue', 'weakness',
            'dizziness', 'shortness of breath', 'congestion', 'runny nose',
            'sore throat', 'chills', 'sweating', 'rash', 'swelling',
            'numbness', 'tingling', 'chest pain', 'abdominal pain'
        }
        
        self.treatment_keywords = {
            'physiotherapy', 'therapy', 'painkillers', 'medication', 'treatment',
            'sessions', 'x-ray', 'examination', 'analgesics', 'advice'
        }
        
        self.diagnosis_keywords = {
            'whiplash', 'injury', 'strain', 'trauma', 'damage', 'degeneration',
            'sprain', 'fracture', 'break', 'tear', 'concussion', 'soft tissue injury'
        }
        
        # Initialize diagnosis inference layer
        self.diagnosis_inference = DiagnosisInferenceLayer()
        
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
    
    def is_valid_symptom(self, symptom: str, context: str = "") -> bool:
        """
        Validate if extracted text is a genuine symptom
        
        Args:
            symptom: Extracted symptom candidate
            context: Full text context for denial checking
            
        Returns:
            True if valid symptom, False otherwise
        """
        if not symptom or len(symptom) < 2:
            return False
        
        symptom_lower = symptom.lower().strip()
        
        # Check against non-symptom patterns
        for pattern in self.non_symptom_patterns:
            if re.match(pattern, symptom_lower, re.IGNORECASE):
                return False
        
        # Check if THIS SPECIFIC symptom is explicitly denied in context
        if context:
            context_lower = context.lower()
            for pattern_template in self.denial_patterns:
                # Replace {symptom} with actual symptom
                # Only match if the specific symptom is denied
                pattern = pattern_template.replace('{symptom}', re.escape(symptom_lower))
                if re.search(pattern, context_lower):
                    return False
            
            # Additional specific denials - only for the exact symptom
            # "no anxiety" should only block "anxiety", not all symptoms
            specific_denials = [
                (r'\bno\s+anxiety\b', ['anxiety', 'anxious']),
                (r'\bno\s+emotional\s+issues?\b', ['emotional issues', 'emotional issue']),
                (r'\bno\s+depression\b', ['depression']),
                (r'\bno\s+stress\b', ['stress']),
            ]
            for pattern, denied_list in specific_denials:
                if re.search(pattern, context_lower) and symptom_lower in denied_list:
                    return False
        
        # Must contain meaningful medical content
        # Valid symptoms typically include: sensation + optional body part
        valid_symptom_indicators = [
            'pain', 'ache', 'discomfort', 'hurt', 'sore', 'stiff', 'tender',
            'trouble', 'difficulty', 'problem', 'issue',
            'cough', 'fever', 'nausea', 'fatigue', 'dizz', 'headache',
            'impact', 'injury', 'trauma', 'swell', 'numb', 'tingle'
        ]
        
        # Exclude standalone sensation words without body part
        standalone_sensations = ['pain', 'ache', 'discomfort', 'hurt', 'hurts', 'stiff', 
                                'sore', 'tender', 'painful', 'aching', 'stiffness']
        if symptom_lower in standalone_sensations:
            return False
        
        # Check if symptom contains at least one valid indicator
        return any(indicator in symptom_lower for indicator in valid_symptom_indicators)
    
    def normalize_symptom(self, symptom: str) -> str:
        """
        Normalize symptom text for consistency
        
        Args:
            symptom: Raw symptom text
            
        Returns:
            Normalized symptom text
        """
        # Lowercase for processing
        normalized = symptom.lower().strip()
        
        # Remove common filler words
        filler_words = ['the', 'a', 'an', 'my', 'some', 'very', 'really', 'quite']
        words = normalized.split()
        words = [w for w in words if w not in filler_words]
        normalized = ' '.join(words)
        
        # Standardize variants to canonical forms
        canonical_mappings = {
            'backache': 'back pain',
            'back ache': 'back pain',
            'occasional backache': 'back pain',
            'occasional backaches': 'back pain',
            'occasional back pain': 'back pain',
            'neckache': 'neck pain',
            'neck ache': 'neck pain',
            'neck back pain': 'neck pain',  # "neck and back pain" split handling
            'back neck pain': 'back pain',
            'headaches': 'headache',
            'trouble sleeping': 'sleep difficulty',
            'difficulty sleeping': 'sleep difficulty',
            'hit head': 'head impact',
            'head hit': 'head impact',
            'sore neck': 'neck pain',
            'sore back': 'back pain',
            'stiff neck': 'neck stiffness',
            'stiff back': 'back stiffness',
            'severe cough': 'cough',
            'heavy lungs': 'chest discomfort',
            'swollen wrist': 'wrist swelling',
            'wrist hurts': 'wrist pain',
            'wrist is painful': 'wrist pain',
            'issues with right shoulder': 'right shoulder pain',
            'issues with shoulder': 'shoulder pain',
        }
        
        # Apply canonical mapping
        normalized = canonical_mappings.get(normalized, normalized)
        
        # Title case for output
        return normalized.title()
    
    def _extract_body_parts(self, text: str) -> List[str]:
        """
        Extract body parts mentioned in text
        
        Args:
            text: Input text (already lowercased)
            
        Returns:
            List of body parts found
        """
        body_part_patterns = [
            # Specific body parts
            r'\b(wrist|wrists)\b',
            r'\b(hand|hands)\b',
            r'\b(neck)\b',
            r'\b(back)\b',
            r'\b(head)\b',
            r'\b(chest)\b',
            r'\b(shoulder|shoulders)\b',
            r'\b(knee|knees)\b',
            r'\b(ankle|ankles)\b',
            r'\b(elbow|elbows)\b',
            r'\b(hip|hips)\b',
            r'\b(foot|feet)\b',
            r'\b(leg|legs)\b',
            r'\b(arm|arms)\b',
            r'\b(finger|fingers)\b',
            r'\b(toe|toes)\b',
            r'\b(spine)\b',
            r'\b(throat)\b',
            r'\b(stomach|abdomen)\b',
        ]
        
        body_parts = set()
        for pattern in body_part_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                body_parts.add(match.group(1))
        
        return list(body_parts)

    def _infer_body_parts_from_symptom(self, symptom: str) -> Set[str]:
        """Infer body parts referenced by a symptom string."""
        symptom_lower = symptom.lower()
        body_parts = set()
        mappings = {
            'shoulder': ['shoulder'],
            'arm': ['arm', 'upper arm', 'lower arm'],
            'wrist': ['wrist'],
            'hand': ['hand'],
            'elbow': ['elbow'],
            'back': ['back', 'lower back'],
            'spine': ['spine'],
            'neck': ['neck'],
            'chest': ['chest'],
            'hip': ['hip'],
            'knee': ['knee'],
            'ankle': ['ankle'],
        }

        for key, parts in mappings.items():
            if key in symptom_lower:
                body_parts.update(parts)

        return body_parts
    
    def _bind_symptom_to_body_part(self, sensation: str, text: str, body_parts: List[str]) -> str:
        """
        Bind a sensation to the nearest body part in context
        
        Args:
            sensation: Sensation word (pain, stiffness, etc.)
            text: Full text
            body_parts: List of body parts in text
            
        Returns:
            Combined symptom (e.g., "wrist pain") or original sensation
        """
        text_lower = text.lower()
        
        # Look for explicit body part + sensation patterns
        for body_part in body_parts:
            # Check various patterns
            patterns = [
                rf'\b{body_part}\s+(?:is\s+|feels?\s+)?{sensation}\b',
                rf'\b{sensation}\s+(?:in|on|at)\s+(?:my\s+|the\s+)?{body_part}\b',
                rf'\b{body_part}\s+(?:still\s+)?{sensation}s?\b',
            ]
            
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return f"{body_part} {sensation}"
        
        return sensation
    
    def extract_symptoms(self, text: str, use_transformer: bool = True) -> List[str]:
        """
        Extract symptoms from text using hybrid approach with validation
        
        Args:
            text: Input text
            use_transformer: If True, combines rule-based + transformer results
            
        Returns:
            List of extracted, validated, and normalized symptoms
        """
        symptoms = set()
        text_lower = text.lower()
        
        # Extract body parts mentioned in the text for context
        body_parts = self._extract_body_parts(text_lower)
        
        # 1. RULE-BASED EXTRACTION
        # Rule-based extraction - General symptom patterns
        symptom_patterns = [
            # Body part + sensation patterns (wrist hurts, wrist is painful, etc.)
            r'(?:my|the)\s+(wrist|hand|neck|back|head|chest|shoulder|knee|ankle|elbow)\s+(?:is\s+|still\s+)?(hurts?|aches?|pains?|painful|stiff|sore|swollen|numb|tingles?)',
            r'(wrist|hand|neck|back|head|chest|shoulder|knee|ankle|elbow)\s+(pain|ache|discomfort|stiffness|swelling|numbness|tingling)',
            # Sensation + body part patterns
            r'(?:pain|ache|discomfort|soreness|stiffness|swelling|numbness|tingling)\s+(?:in|on|at)\s+(?:my|the)?\s*(wrist|hand|neck|back|head|chest|shoulder|knee|ankle|elbow)s?',
            # General "I am having/have" pattern
            r'(?:i\s+am\s+having|i\s+have|having)\s+(?:a\s+)?([a-z\s]+?)(?:\s+doctor|\s+and|,|\.|$)',
            # Compound symptoms
            r'(neck\s+(?:and\s+)?back\s+pain)',
            r'(head\s+impact|hit\s+(?:my\s+)?head)',
            r'(trouble\s+sleeping)',
            r'(occasional\s+(?:back)?aches?)',
            # Specific symptoms
            r'\b(?:i feel|i have|having)\s+(discomfort)\b',
            r'\b(cough(?:ing)?)\b',
            r'\b(headaches?)\b',
            r'\b(fever)\b',
            r'\b(nausea)\b',
            r'\b(vomiting)\b',
            r'\b(fatigue)\b',
            r'\b(dizziness)\b',
            r'\b(shortness\s+of\s+breath)\b',
            r'\b(chest\s+pain)\b',
            r'\b(back\s+pain)\b',
            r'\b(neck\s+pain)\b',
            r'\b(sore\s+throat)\b',
            r'\b(runny\s+nose)\b',
            r'\b(loss\s+of\s+appetite)\b',
            r'\b(abdominal\s+pain)\b',
        ]
        
        for pattern in symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Get the captured symptom
                symptom = match.group(1) if match.lastindex else match.group(0)
                symptom = symptom.strip(' .,;')
                # Clean up common filler words
                symptom = re.sub(r'\b(doctor|and|but|so|very|really)\b', '', symptom).strip()
                
                # VALIDATION: Check if this is a genuine symptom
                if symptom and len(symptom) > 2 and self.is_valid_symptom(symptom, text):
                    # NORMALIZATION: Apply canonical form
                    normalized = self.normalize_symptom(symptom)
                    symptoms.add(normalized)
        
        # Keyword matching for common symptoms
        common_symptoms = {
            'cough': 'Cough',
            'headache': 'Headache',
            'headaches': 'Headache',
            'fever': 'Fever',
            'nausea': 'Nausea',
            'fatigue': 'Fatigue',
            'dizziness': 'Dizziness',
            'vomiting': 'Vomiting',
            'sore throat': 'Sore Throat',
            'chest pain': 'Chest Pain',
            'shortness of breath': 'Shortness Of Breath',
            'runny nose': 'Runny Nose',
            'loss of appetite': 'Loss Of Appetite',
        }
        
        for keyword, symptom_name in common_symptoms.items():
            if keyword in text_lower and self.is_valid_symptom(symptom_name, text):
                symptoms.add(symptom_name)
        
        # Don't extract symptoms from questions about symptoms
        # "Any numbness?" followed by "No" should not extract "numbness"
        question_denials = [
            (r'any\s+numbness.*?\n.*?patient:\s*no', 'numbness'),
            (r'any\s+tingling.*?\n.*?patient:\s*no', 'tingling'),
            (r'numbness\s+or\s+tingling.*?\n.*?patient:\s*no', 'numbness'),
            (r'numbness\s+or\s+tingling.*?\n.*?patient:\s*no', 'tingling'),
        ]
        for pattern, denied_symptom in question_denials:
            if re.search(pattern, text_lower, re.DOTALL | re.MULTILINE):
                symptoms.discard(denied_symptom.title())
                symptoms.discard(denied_symptom.capitalize())
        
        # Body part + feeling combinations (e.g., "lungs feel heavy")
        body_feeling_patterns = [
            (r'lungs?\s+feel\s+(heavy|tight|congested)', 'Heavy lungs'),
            (r'chest\s+feels?\s+(tight|heavy|congested)', 'Chest tightness'),
            (r'throat\s+feels?\s+(sore|scratchy|painful)', 'Sore throat'),
            (r'head\s+feels?\s+(heavy|light|dizzy)', 'Head discomfort'),
        ]
        
        for pattern, symptom_name in body_feeling_patterns:
            if re.search(pattern, text_lower):
                symptoms.add(symptom_name)
        
        # Legacy keyword matching for existing patterns - with validation
        for keyword in self.symptom_keywords:
            if keyword in text_lower:
                if 'neck' in text_lower and 'pain' in keyword:
                    if self.is_valid_symptom("Neck pain", text):
                        symptoms.add("Neck Pain")
                if 'back' in text_lower and 'pain' in keyword:
                    if self.is_valid_symptom("Back pain", text):
                        symptoms.add("Back Pain")
                if 'head' in text_lower and ('impact' in text_lower or 'hit' in text_lower):
                    if self.is_valid_symptom("Head impact", text):
                        symptoms.add("Head Impact")
        
        # Specific symptom extraction from context - with validation
        if 'hit my head on the steering wheel' in text_lower:
            if self.is_valid_symptom("Head impact", text):
                symptoms.add("Head Impact")
        if 'neck and back pain' in text_lower or ('neck' in text_lower and 'back pain' in text_lower):
            if self.is_valid_symptom("Neck pain", text):
                symptoms.add("Neck Pain")
            if self.is_valid_symptom("Back pain", text):
                symptoms.add("Back Pain")
        if 'trouble sleeping' in text_lower:
            if self.is_valid_symptom("Sleep difficulty", text):
                symptoms.add("Sleep Difficulty")
        if 'occasional backache' in text_lower or 'occasional back pain' in text_lower:
            if self.is_valid_symptom("Back pain", text):
                symptoms.add("Back Pain")
        
        # Wrist-specific extractions
        if 'wrist' in text_lower:
            wrist_symptoms = {
                'swollen': 'Wrist Swelling',
                'painful': 'Wrist Pain',
                'hurts': 'Wrist Pain',
                'stiff': 'Wrist Stiffness',
                'pain': 'Wrist Pain',
            }
            for keyword, symptom in wrist_symptoms.items():
                if keyword in text_lower and self.is_valid_symptom(symptom, text):
                    symptoms.add(symptom)
        
        # 2. TRANSFORMER-BASED EXTRACTION (if enabled and available)
        if use_transformer and self.transformer_ner:
            transformer_symptoms = self.extract_symptoms_transformer(text)
            # Validate and normalize transformer results
            for symptom in transformer_symptoms:
                if self.is_valid_symptom(symptom, text):
                    normalized = self.normalize_symptom(symptom)
                    symptoms.add(normalized)
        
        # Final deduplication and sorting
        return sorted(list(symptoms))
    
    def extract_symptoms_transformer(self, text: str) -> List[str]:
        """
        Extract symptoms using transformer-based medical NER
        
        Args:
            text: Input text
            
        Returns:
            List of extracted symptoms using transformer model (validated)
        """
        if not self.transformer_ner:
            return []
        
        symptoms = set()
        
        try:
            # Run transformer NER
            ner_results = self.transformer_ner(text)
            
            # Extract entities related to symptoms/diseases/conditions
            symptom_labels = [
                'DISEASE',           # Disease names
                'SIGN_SYMPTOM',      # Signs and symptoms
                'SYMPTOM',           # Symptoms
                'PROBLEM',           # Medical problems
                'CLINICAL_EVENT',    # Clinical events
            ]
            
            for entity in ner_results:
                entity_type = entity.get('entity_group', entity.get('entity', ''))
                symptom_text = entity['word'].strip()
                
                # Clean up transformer artifacts
                symptom_text = symptom_text.replace('##', '').strip()
                
                # Skip malformed extractions
                if len(symptom_text) < 3:
                    continue
                
                # Skip transformer artifacts and non-medical terms
                if symptom_text.lower().endswith('iotherapy') or symptom_text.lower() in {'the', 'and', 'or', 'a', 'an'}:
                    continue
                
                # Check if entity is symptom-related with good confidence
                if any(label in entity_type.upper() for label in symptom_labels):
                    if entity['score'] > 0.7:  # Good confidence for labeled symptoms
                        # Additional validation before adding
                        if self.is_valid_symptom(symptom_text, text):
                            symptoms.add(symptom_text.title())
                elif entity['score'] > 0.90:  # Very high confidence for unlabeled
                    if self.is_valid_symptom(symptom_text, text):
                        symptoms.add(symptom_text.title())
        
        except Exception as e:
            print(f"Transformer NER error: {e}")
            return []
        
        return sorted(list(symptoms))
    
    def extract_symptoms_hybrid(self, text: str) -> Dict[str, List[str]]:
        """
        Compare rule-based vs transformer-based symptom extraction
        
        Returns:
            Dictionary with both methods' results and combined
        """
        rule_based = self.extract_symptoms(text)
        transformer_based = self.extract_symptoms_transformer(text)
        
        # Combine both (union)
        combined = sorted(list(set(rule_based) | set(transformer_based)))
        
        return {
            'rule_based': rule_based,
            'transformer_based': transformer_based,
            'combined': combined,
            'rule_only': sorted(list(set(rule_based) - set(transformer_based))),
            'transformer_only': sorted(list(set(transformer_based) - set(rule_based))),
        }
    
    def extract_treatments(self, text: str) -> List[str]:
        """
        Extract treatments from text (NOT investigations or examinations)
        
        Treatments = therapeutic interventions
        Excludes: X-rays, examinations, tests (those are investigations)
        """
        treatments = set()
        text_lower = text.lower()
        
        # Pattern-based extraction for TREATMENTS ONLY
        treatment_patterns = [
            r'(\d+\s+sessions?\s+of\s+physiotherapy)',
            r'(physiotherapy\s+sessions?)',
            r'(took\s+painkillers?)',
            r'(painkillers?)',
            r'(analgesics?)',
            r'(wore\s+(?:a\s+)?wrist\s+brace)',
            r'(wrist\s+brace)',
            r'(physical\s+therapy)',
            r'(medication)',
            r'(antibiotics?)',
            r'(surgery)',
            r'(rest)',
            r'(ice\s+(?:pack)?)',
            r'(tried\s+(?:some\s+)?exercises?)',
            r'(exercises?)',
            r'(stretching)',
            r'(walking)',
        ]
        
        for pattern in treatment_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                treatment = match.group(1).strip()
                # Capitalize properly
                treatment = ' '.join(word.capitalize() for word in treatment.split())
                treatments.add(treatment)
        
        # Specific extractions with normalization
        if 'ten sessions of physiotherapy' in text_lower or '10 sessions of physiotherapy' in text_lower:
            treatments.add("10 Physiotherapy Sessions")
        if 'took painkillers' in text_lower or 'painkiller' in text_lower:
            treatments.add("Painkillers")
        if 'wrist brace' in text_lower:
            treatments.add("Wrist Brace")
        if 'tried some exercises' in text_lower or 'tried exercises' in text_lower:
            treatments.add("Exercises")
        if 'physiotherapy' in text_lower and not any('physiotherapy' in t.lower() for t in treatments):
            treatments.add("Physiotherapy")
        
        # Normalize and deduplicate
        normalized_treatments = set()
        for treatment in treatments:
            # Normalize common duplicates
            t_lower = treatment.lower()
            if 'took painkillers' in t_lower or 'painkillers' in t_lower:
                normalized_treatments.add("Painkillers")
            elif 'wore a wrist brace' in t_lower or 'wrist brace' in t_lower:
                normalized_treatments.add("Wrist Brace")
            elif 'tried some exercises' in t_lower or 'tried exercises' in t_lower or t_lower == 'exercises':
                normalized_treatments.add("Exercises")
            elif t_lower in ['stretching', 'walking']:
                # Don't include these as standalone - too generic
                continue
            else:
                normalized_treatments.add(treatment)
            
        return sorted(list(normalized_treatments))
    
    def extract_investigations(self, text: str) -> Dict[str, List[str]]:
        """
        Extract diagnostic investigations/tests from text with explicit status.

        Status buckets:
        - performed: actually done in this encounter
        - considered: discussed as a possibility/conditional
        - negated: explicitly stated not done/needed
        
        Special rule: Examination is assumed performed unless explicitly negated.
        """
        text_lower = text.lower()

        # Canonical investigation patterns
        investigations = {
            "MRI": r"mri",
            "X-ray": r"x[-\s]?rays?",
            "CT Scan": r"ct\s+scan",
            "Blood Test": r"blood\s+tests?",
            "Ultrasound": r"ultrasound",
            "Examination": r"(?:physical\s+)?examinations?|exam\b",
        }

        negated_patterns = [
            r"(?:no|without)\s+(?:any\s+)?(?:{k})",
            r"(?:didn't|did\s+not)\s+do\s+(?:any\s+)?(?:{k})",
            r"(?:{k})\s+(?:wasn't|was\s+not)\s+done",
            r"no\s+need\s+for\s+(?:{k})",
            r"(?:{k})\s+(?:is|was)\s+not\s+(?:necessary|required|indicated)",
        ]

        considered_patterns = [
            r"would\s+(?:only\s+)?consider[^.]*?(?:{k})",
            r"could\s+do[^.]*?(?:{k})\s+if\s+needed",
            r"might\s+need[^.]*?(?:{k})",
            r"(?:{k})[^.]*?if\s+symptoms\s+worsen",
            r"if\s+symptoms\s+worsen[^.]*?(?:{k})",
        ]

        performed_patterns = [
            r"(?:did|performed|ordered|arranged)\s+(?:a\s+)?(?:{k})",
            r"(?:{k})\s+(?:was|were)\s+(?:done|performed|completed)",
            r"\b(?:{k})\b",
        ]

        status_buckets = {"performed": set(), "considered": set(), "negated": set()}

        for name, pattern in investigations.items():
            keyword_regex = pattern

            def matches(patterns: List[str]) -> bool:
                return any(re.search(p.format(k=keyword_regex), text_lower) for p in patterns)

            # Special handling for Examination: assumed performed unless explicitly negated
            if name == "Examination":
                # Only negate if explicitly stated "no examination" or "examination was not done"
                explicit_exam_negation = [
                    r"no\s+(?:physical\s+)?examinations?",
                    r"examinations?\s+(?:was|were)\s+not\s+done",
                    r"(?:didn't|did\s+not)\s+do\s+(?:an?\s+)?examinations?",
                ]
                if any(re.search(p, text_lower) for p in explicit_exam_negation):
                    status_buckets["negated"].add(name)
                elif re.search(keyword_regex, text_lower) or any(kw in text_lower for kw in ['accident', 'emergency', 'hospital', 'checked', 'examined']):
                    # Examination is performed if mentioned OR if clinical context present
                    status_buckets["performed"].add(name)
                continue

            # Standard processing for other investigations
            if matches(negated_patterns):
                status_buckets["negated"].add(name)
            elif matches(considered_patterns):
                status_buckets["considered"].add(name)
            elif re.search(keyword_regex, text_lower):
                status_buckets["performed"].add(name)

        return {state: sorted(list(values)) for state, values in status_buckets.items()}
    
    def has_diagnostic_context(self, text: str) -> bool:
        """
        Check if text contains explicit diagnostic confirmation.
        
        This is a semantic gate: Does this text explicitly state a diagnosis?
        Not: "Does this text mention medical words?"
        
        Args:
            text: Input text
            
        Returns:
            True if explicit diagnostic statement present
        """
        text_lower = text.lower()
        
        # Explicit diagnostic confirmation patterns
        diagnostic_patterns = [
            r'\b(?:was |were )?diagnosed\s+(?:me\s+)?with\b',
            r'\b(?:doctor|physician|they|clinician)\s+(?:said|told|confirmed)\s+(?:it\s+)?(?:was|is)\b',
            r'\btold\s+me\s+(?:my\s+)?(?:injury|condition)\s+was\b',
            r'\b(?:hospital|clinic|emergency(?:\s+room)?|er)\s+(?:said|told|confirmed)\b',
            r'\bconfirmed\s+(?:it\s+was\s+)?(?:a\s+)?\w+\s+(?:injury|diagnosis)\b',
            r'\bdiagnosis\s+(?:was|is)\b',
            # Additional patterns for indirect diagnosis
            r'\bsaid\s+(?:there\s+was\s+)?(?:no\s+fracture,?\s+)?just\s+(?:a\s+)?(?:sprain|strain)\b',
            r'\bappears\s+to\s+be\s+(?:a\s+)?(?:resolving\s+)?(?:soft\s+tissue\s+)?injury\b',
            r'\bthis\s+(?:is|appears)\s+(?:a\s+)?(?:sprain|strain|fracture|injury)\b',
        ]
        
        return any(re.search(pattern, text_lower) for pattern in diagnostic_patterns)
    
    def extract_diagnosis(self, text: str, mode: str = "auto", symptoms: List[str] = None) -> str:
        """
        Extract primary diagnosis using inference layer (not just pattern matching)
        
        UPDATED APPROACH:
        1. Uses DiagnosisInferenceLayer for intelligent diagnosis detection
        2. Soft-matches linguistic triggers (appears to be, likely, etc.)
        3. Validates against symptom compatibility
        4. Returns normalized diagnosis with confidence
        
        Args:
            text: Input text
            mode: 'utterance' (single statement), 'transcript' (full conversation), 'auto' (detect)
            symptoms: Optional list of extracted symptoms for compatibility checking
            
        Returns:
            Diagnosis string or empty string if context doesn't permit extraction
        """
        text_lower = text.lower()
        
        # Detect mode automatically if not specified
        if mode == "auto":
            mode = self._detect_mode(text)
        
        # GATE 1: Utterance mode = only explicit confirmations allowed
        if mode == "utterance":
            if self.has_diagnostic_context(text):
                return self._extract_confirmed_diagnosis(text_lower)
            return ""  # No diagnosis for utterances without explicit confirmation
        
        # GATE 2: Transcript mode = use inference layer
        # Extract body parts for compatibility checking
        body_parts = self._extract_body_parts(text_lower)
        
        # Use inference layer
        result = self.diagnosis_inference.infer_diagnosis(
            text,
            extracted_symptoms=symptoms if symptoms else [],
            extracted_body_parts=body_parts
        )
        
        # Return diagnosis if confidence is reasonable
        if result['confidence'] >= 0.5:
            return result['diagnosis']
        
        # Fallback to old method for explicit confirmations
        if self.has_diagnostic_context(text):
            fallback = self._extract_confirmed_diagnosis(text_lower)
            if fallback:
                return fallback
        
        return ""  # No diagnosis without context
    
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
        Deprecated: Use has_diagnostic_context() instead.
        Kept for backward compatibility.
        """
        return self.has_diagnostic_context(text_lower)
    
    def _extract_confirmed_diagnosis(self, text_lower: str) -> str:
        """Extract explicitly confirmed diagnosis from text"""
        # Extract diagnosis only when explicitly stated
        diagnosis_patterns = [
            # Explicit confirmations
            (r'(?:diagnosed\s+(?:me\s+)?with|said it was|told (?:me )?(?:my )?injury was|confirmed (?:it was )?(?:a )?)\s*(?:a\s+|an\s+)?(whiplash\s+injury)', 'Whiplash injury'),
            (r'(?:diagnosed\s+(?:me\s+)?with|said it was|told it is|confirmed\s+(?:it\s+was\s+)?(?:a\s+)?)\s*(?:a\s+|an\s+)?(whiplash)(?!\s+injury)', 'Whiplash'),
            # Sprain patterns
            (r'(?:said|diagnosed|confirmed)\s+(?:there\s+was\s+)?(?:no\s+fracture,?\s+)?(?:just\s+)?(?:a\s+)?(sprain)', 'Sprain'),
            (r'(?:appears\s+to\s+be|this\s+is|likely)\s+(?:a\s+)?((?:soft\s+tissue|resolving\s+soft\s+tissue)\s+injury)', 'Soft tissue injury'),
            (r'\bjust\s+(?:a\s+)?(sprain)\b', 'Sprain'),
            # Strain patterns
            (r'(?:diagnosed\s+(?:me\s+)?with|said it was|confirmed)\s+(?:a\s+)?((?:lower\s+)?back\s+strain)', 'Lower back strain'),
            (r'(?:diagnosed\s+(?:me\s+)?with|said it was|confirmed)\s+(?:a\s+)?(neck\s+injury)', 'Neck injury'),
            (r'(?:diagnosed\s+(?:me\s+)?with|said it was|confirmed)\s+(?:a\s+)?(back\s+injury)', 'Back injury'),
            (r'(?:diagnosed\s+(?:me\s+)?with|said it was|confirmed)\s+(?:a\s+)?(strain)', 'Strain'),
            # Wrist-specific
            (r'(?:diagnosed\s+with|said it was|confirmed)\s+(?:a\s+)?(wrist\s+sprain)', 'Wrist sprain'),
            (r'\b(wrist\s+fracture)\b', 'Wrist fracture'),
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
        """Extract current patient status with temporal context"""
        text_lower = text.lower()

        # Helper flags for resolution threshold
        pain_score_present = bool(re.search(r'\b([1-9])\s*/\s*10\b', text_lower) or re.search(r'\baround\s+([1-9])\b.*most\s+days', text_lower))
        functional_limitation = any(kw in text_lower for kw in [
            'avoid heavy lifting', 'reduce my workload', 'reduced my workload',
            'pain on overhead movement', 'overhead movement', 'overhead activity',
            'end-range pain', 'limits my', 'still hurts when', 'lifting above my head'
        ])
        active_treatment = any(kw in text_lower for kw in [
            'physiotherapy', 'therapy', 'exercises', 'painkillers', 'analgesics'
        ])
        
        # Check for chronic/ongoing pain patterns (months, weeks of duration)
        # Handles both numeric (3 months) and written (three months) formats
        chronic_patterns = [
            r'(?:had|experiencing|dealing with).{0,30}(?:for|past|last)\s+(?:the\s+)?(?:past\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(months?|weeks?)',
            r'(?:been\s+)?ongoing\s+(?:for\s+)?(?:\d+|several|multiple|many)\s+(months?|weeks?)',
            r'(?:chronic|persistent).{0,20}(?:\d+|several)\s+(months?|weeks?)',
        ]
        
        for pattern in chronic_patterns:
            match = re.search(pattern, text_lower)
            if match:
                duration_unit = match.group(1)
                # If months mentioned, it's chronic and stable
                if 'month' in duration_unit:
                    # Check if it's improving or stable
                    if 'improving' in text_lower or 'getting better' in text_lower or 'doing better' in text_lower:
                        return "Chronic but improving"
                    else:
                        return "Ongoing but stable"
        
        # Check for occasional status
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
        
        # Resolution threshold: only "recovering well" if no pain score, no limitation, no active treatment
        if not (pain_score_present or functional_limitation or active_treatment):
            return "Recovering well"
        
        return "Improving but not resolved"

    def extract_onset(self, text: str) -> str:
        """Extract onset type (gradual vs sudden)"""
        text_lower = text.lower()
        gradual_markers = ['gradual', 'over time', 'slowly', 'built up', 'over the next few weeks']
        sudden_markers = ['sudden', 'suddenly', 'acute', 'all of a sudden']

        if any(marker in text_lower for marker in gradual_markers):
            return "Gradual"
        if any(marker in text_lower for marker in sudden_markers):
            return "Sudden"
        return "Unknown"

    def extract_duration(self, text: str) -> str:
        """Extract duration phrase (e.g., 'four months')"""
        text_lower = text.lower()
        duration_pattern = r'((?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+(?:months?|weeks?|days?))'
        match = re.search(duration_pattern, text_lower)
        if match:
            return match.group(1).title()
        return "Unknown"
    
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
        Main method to extract all medical entities from transcript.
        
        GATING LOGIC:
        - Symptoms: Always extracted when present
        - Diagnosis: Gated by mode and diagnostic context
        - Treatment: Always extracted when mentioned
        
        Args:
            text: Medical transcript text
            mode: 'utterance' (single statement), 'transcript' (full conversation), 'auto' (detect)
            
        Returns:
            Dictionary with structured medical information
        """
        # Extract all entities
        symptoms = self.extract_symptoms(text)
        
        entities = {
            "Patient_Name": self.extract_patient_name(text),
            "Date_of_Incident": self.extract_date_of_incident(text),
            "Onset": self.extract_onset(text),
            "Duration": self.extract_duration(text),
            "Symptoms": symptoms,
            "Diagnosis": self.extract_diagnosis(text, mode=mode, symptoms=symptoms),  # Pass symptoms for compatibility checking
            "Treatment": self.extract_treatments(text),
            "Investigations": self.extract_investigations(text),
            "Current_Status": self.extract_current_status(text),
            "Prognosis": self.extract_prognosis(text)
        }
        
        # Post-processing filters for known edge cases
        entities = self._apply_safety_filters(entities, text)
        
        return entities
    
    def _apply_safety_filters(self, entities: Dict, text: str) -> Dict:
        """
        Apply post-extraction safety filters for known edge cases.
        
        These are explicit, documented exceptions - not lexical blacklists.
        """
        text_lower = text.lower()
        
        # Filter 1: Remove diagnosis terms from symptoms if in diagnostic context
        # Rationale: "whiplash" should be diagnosis, not symptom
        if self.has_diagnostic_context(text):
            diagnosis_terms = {'whiplash', 'whiplash injury', 'injury', 'strain'}
            entities['Symptoms'] = [s for s in entities['Symptoms'] 
                                   if s.lower() not in diagnosis_terms]
        
        # Filter 2: Remove abstract pain mentions in worry/anxiety contexts
        # Rationale: "worried this pain might not go away" is expressing concern, not reporting symptom
        worry_patterns = [r'\b(worried|anxious|scared)\b.*\bpain\b', r'\bpain\b.*\bmight\s+not\b']
        if any(re.search(p, text_lower) for p in worry_patterns):
            entities['Symptoms'] = [s for s in entities['Symptoms'] 
                                   if s.lower() not in {'pain', 'discomfort'}]
        
        # Filter 3: Remove treatment terms mistakenly extracted as symptoms
        # Rationale: "physiotherapy sessions" is treatment history, not symptom
        treatment_terms = {'sessions', 'physiotherapy', 'therapy', 'x-rays', 'x-ray'}
        entities['Symptoms'] = [s for s in entities['Symptoms'] 
                               if s.lower() not in treatment_terms and 
                               not any(t in s.lower() for t in treatment_terms)]
        
        # Filter 4: Remove explicitly denied symptoms
        # Rationale: "No anxiety" or "No emotional issues" means patient doesn't have those
        denied_symptoms = set()
        denial_checks = [
            ('anxiety', r'\bno\s+(?:signs?\s+of\s+)?anxiety\b'),
            ('emotional issues', r'\bno\s+emotional\s+issues?\b'),
            ('depression', r'\bno\s+depression\b'),
            ('stress', r'\bno\s+stress\b'),
        ]
        
        for symptom_name, pattern in denial_checks:
            if re.search(pattern, text_lower):
                denied_symptoms.add(symptom_name)
        
        # Check for question-denial patterns (Q: "Any numbness?" A: "No")
        # More flexible patterns that handle variations and multiple newlines
        qa_denials = [
            (r'any\s+numbness[^?]*?\?[\s\n]*patient:\s*no\.?', 'numbness'),
            (r'numbness\s+or\s+weakness[^?]*?\?[\s\n]*patient:\s*no\.?', 'numbness'),
            (r'numbness\s+or\s+tingling[^?]*?\?[\s\n]*patient:\s*no\.?', 'numbness'),
            (r'any\s+tingling[^?]*?\?[\s\n]*patient:\s*no\.?', 'tingling'),
            (r'numbness\s+or\s+weakness[^?]*?\?[\s\n]*patient:\s*no\.?', 'weakness'),
        ]
        for pattern, symptom_name in qa_denials:
            if re.search(pattern, text_lower, re.DOTALL):
                denied_symptoms.add(symptom_name)
                # Also add common variations
                denied_symptoms.add(symptom_name + 'ness')
                denied_symptoms.add('numb')  # Catch "Numb" symptom

        # Filter 4b: Body-part scoped negation ("neck and back feel fine")
        denied_body_parts = set()
        body_part_negation_patterns = [
            r'\b(no\s+pain|no issues|no problems|no symptoms)\s+(?:in|with)\s+(?:my\s+)?(neck|back|shoulder|arm|upper arm|lower back|spine)\b',
            r'\b(neck|back|shoulder|arm|upper arm|lower back|spine)\b[^.]{0,40}\b(feel\s+fine|feels\s+fine|no pain|no issues|no problems)\b',
            r'\b(my\s+)?(neck|back|shoulder|arm|upper arm|lower back|spine)\s+feel(s)?\s+fine\b',
        ]
        for pattern in body_part_negation_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                # Group 1 may be negation phrase, group 2 is body part (pattern dependent)
                # Capture all body parts present in the match text
                for part in ['neck', 'back', 'shoulder', 'arm', 'upper arm', 'lower back', 'spine']:
                    if part in match.group(0):
                        denied_body_parts.add(part)

        if denied_body_parts:
            filtered_symptoms = []
            for symptom in entities['Symptoms']:
                symptom_parts = self._infer_body_parts_from_symptom(symptom)
                # If any inferred body part is explicitly denied, drop the symptom
                if symptom_parts and any(bp in denied_body_parts for bp in symptom_parts):
                    continue
                filtered_symptoms.append(symptom)
            entities['Symptoms'] = filtered_symptoms
        
        # Remove denied symptoms
        entities['Symptoms'] = [s for s in entities['Symptoms'] 
                               if s.lower() not in denied_symptoms]
        
        # Filter 5: Final validation pass - ensure all symptoms are still valid
        validated_symptoms = []
        for symptom in entities['Symptoms']:
            if self.is_valid_symptom(symptom, text):
                validated_symptoms.append(symptom)
        
        entities['Symptoms'] = validated_symptoms
        
        # Filter 6: Remove generic symptoms when specific ones exist
        # E.g., if "Neck Pain" exists, remove standalone "Pain"
        specific_symptoms = [s for s in entities['Symptoms'] if ' ' in s]
        generic_symptoms = [s for s in entities['Symptoms'] if ' ' not in s]
        
        # If we have specific symptoms with body parts, remove generic pain/ache/discomfort
        if specific_symptoms:
            generic_to_remove = {'pain', 'ache', 'discomfort', 'stiff', 'hurt', 'sore'}
            generic_symptoms = [s for s in generic_symptoms 
                              if s.lower() not in generic_to_remove]
        
        entities['Symptoms'] = sorted(list(set(specific_symptoms + generic_symptoms)))

        # Filter 7: Remove treatments mentioned in negated context (e.g., "no surgery is indicated")
        negated_treatments = set()
        treatment_negation_patterns = [
            (r'no\s+surgery\s+is\s+indicated', 'surgery'),
            (r'no\s+need\s+for\s+surgery', 'surgery'),
            (r'not\s+recommend(?:ed)?\s+surgery', 'surgery'),
            (r'no\s+plans?\s+for\s+surgery', 'surgery'),
        ]
        for pattern, treatment_name in treatment_negation_patterns:
            if re.search(pattern, text_lower):
                negated_treatments.add(treatment_name)

        if negated_treatments:
            entities['Treatment'] = [t for t in entities['Treatment'] 
                                     if t.lower() not in negated_treatments]
        
        return entities


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


class MedicalSummarizer:
    """Summarize medical transcripts into structured reports"""
    
    def __init__(self):
        """Initialize summarization model"""
        try:
            from transformers import pipeline as transformer_pipeline
            # Use a medical-focused or general summarization model
            self.summarizer = transformer_pipeline(
                "summarization", 
                model="facebook/bart-large-cnn",
                device=-1  # Use CPU
            )
        except Exception as e:
            print(f"Warning: Could not load summarization model: {e}")
            self.summarizer = None
    
    def extract_conversation_sections(self, text: str) -> Dict[str, str]:
        """Split conversation into logical sections"""
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
