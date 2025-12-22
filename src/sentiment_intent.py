"""
Sentiment Analysis and Intent Detection Module
Analyzes patient emotions and communication intents.
"""

from typing import Dict, List, Tuple
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """Analyze sentiment in patient dialogues"""
    
    def __init__(self):
        """Initialize sentiment analysis model"""
        self.sentiment_labels = {
            'POSITIVE': 'Reassured',
            'NEGATIVE': 'Anxious',
            'NEUTRAL': 'Neutral'
        }
        
        try:
            # Use a pre-trained sentiment model
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU
            )
        except Exception as e:
            print(f"Warning: Could not load sentiment model: {e}")
            self.sentiment_pipeline = None
    
    def analyze_sentiment(self, text: str) -> str:
        """
        Analyze sentiment of text
        
        Args:
            text: Patient dialogue text
            
        Returns:
            Sentiment label: 'Anxious', 'Neutral', or 'Reassured'
        """
        # Check for neutral historical statements first
        text_lower = text.lower()
        neutral_patterns = [
            r'\b(happened|went|had|took|checked|was|were|did)\b',
            r'\blast\s+(week|month|year|september|january|february|march|april|may|june|july|august|october|november|december)\b',
            r'\b(sessions?|hospital|doctor|physiotherapy|work|after)\b',
            r'\b(diagnosed|said|told|confirmed)\b.*\b(was|is|injury|whiplash)\b',
            r'\bwas\s+(?:not|no)\s+',
            r'\b(?:not|no)\s+(?:required|fracture|serious)\b',
        ]
        
        neutral_matches = sum(1 for pattern in neutral_patterns if re.search(pattern, text_lower))
        if neutral_matches >= 2:
            return 'Neutral'
        
        if not self.sentiment_pipeline:
            return self._rule_based_sentiment(text)
        
        try:
            result = self.sentiment_pipeline(text)[0]
            label = result['label']
            score = result['score']
            
            # Map to medical context
            if label == 'POSITIVE' and score > 0.7:
                return 'Reassured'
            elif label == 'NEGATIVE' and score > 0.7:
                return 'Anxious'
            else:
                # Check for worry/concern keywords
                if self._has_anxiety_keywords(text):
                    return 'Anxious'
                elif self._has_positive_keywords(text):
                    return 'Reassured'
                return 'Neutral'
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> str:
        """Fallback rule-based sentiment analysis"""
        text_lower = text.lower()
        
        # Neutral/informational indicators (check first)
        neutral_patterns = [
            r'\b(happened|went|had|took|checked|was|were|did)\b',
            r'\blast\s+(week|month|year|september|january|february|march|april|may|june|july|august|october|november|december)\b',
            r'\b(sessions?|hospital|doctor|physiotherapy|work)\b',
            r'^\w+\s+(happened|went|had|took|was|were)',
            r'\b(diagnosed|said|told|confirmed)\b.*\b(was|is|injury|whiplash)\b',
            r'\bwas\s+(?:not|no)\s+',
            r'\b(?:not|no)\s+(?:required|fracture|serious)\b',
        ]
        
        neutral_matches = sum(1 for pattern in neutral_patterns if re.search(pattern, text_lower))
        if neutral_matches >= 2:
            return 'Neutral'
        
        # Anxiety indicators
        anxiety_words = [
            'worried', 'concern', 'anxious', 'nervous', 'scared', 'afraid',
            'fear', 'stress', 'trouble', 'bad', 'worse', 'difficult'
        ]
        
        # Positive indicators
        positive_words = [
            'better', 'good', 'great', 'relief', 'thank', 'appreciate',
            'improving', 'hope', 'glad', 'happy', 'encouraging'
        ]
        
        anxiety_count = sum(1 for word in anxiety_words if word in text_lower)
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        if anxiety_count > positive_count:
            return 'Anxious'
        elif positive_count > anxiety_count:
            return 'Reassured'
        else:
            return 'Neutral'
    
    def _has_anxiety_keywords(self, text: str) -> bool:
        """Check for anxiety keywords"""
        anxiety_keywords = [
            'worried', 'concern', 'anxious', 'nervous', 'afraid',
            'scared', 'worry', 'fear', 'stress'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in anxiety_keywords)
    
    def _has_positive_keywords(self, text: str) -> bool:
        """Check for positive keywords"""
        positive_keywords = [
            'better', 'good', 'great', 'relief', 'thank',
            'hope', 'glad', 'happy', 'improving'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in positive_keywords)


class IntentDetector:
    """Detect patient communication intent"""
    
    def __init__(self):
        """Initialize intent detection"""
        # Pattern priority matters: earlier categories are checked first
        self.intent_patterns = {
            "Answering questions": [
                r'^(?:yes|no)(?:\s|,|\.)',
                r'^(?:yeah|yep|nope|nah)(?:\s|,|\.)',
                r'^(?:i\s+(?:do|did|was|am|have|had|went|always|never))\b',
                r'^(?:nothing|not\s+really|not\s+at\s+all)(?:\s+like\s+that)?\.?$',
            ],
            "Seeking reassurance": [
                r'(?:will|can)\s+(?:i|this)\s+(?:get\s+)?better',
                r'(?:should|do)\s+i\s+(?:be\s+)?worr(?:y|ied)',
                r'is\s+(?:this|it)\s+(?:going\s+to\s+)?(?:be\s+)?(?:okay|fine|normal)',
                r'will\s+(?:this|it)\s+affect',
                r'(?:hope|hoping)\s+(?:it|this)\s+(?:gets?\s+)?better',
            ],
            "Reporting symptoms": [
                r'i\s+(?:have|had|feel|felt)\s+(?:a\s+)?(?:pain|ache|discomfort|hurt)',
                r'my\s+\w+\s+(?:hurts?|aches?|pains?)',
                r'(?:it|pain|discomfort)\s+(?:was|is)\s+(?:really\s+)?(?:bad|severe|intense)',
                r'i\s+(?:can|could)\s+feel\s+(?:pain|discomfort)',
                r'(?:it\'?s\s+)?not\s+constant.*(?:get|have)\s+(?:occasional|some)',
                r'^(?:at\s+first|initially|in\s+the\s+beginning)',
                r'(?:weeks?|months?)\s+(?:were|was)\s+(?:rough|bad|difficult|painful)',
            ],
            "Describing treatment history": [
                r'i\s+(?:had|went\s+to|received|took)\s+(?:\w+\s+)?(?:sessions?|therapy|treatment|medication)',
                r'they\s+(?:gave|prescribed|recommended)',
                r'i\s+(?:had|did)\s+(?:ten|10|\d+)\s+sessions?',
            ],
            "Expressing concern": [
                r'(?:i\'m|i\s+am)\s+(?:a\s+)?(?:bit\s+)?(?:worried|concerned|anxious)',
                r'(?:what|how)\s+(?:about|if)',
                r'(?:can|will)\s+this\s+(?:cause|lead\s+to)',
            ],
            "Asking questions": [
                r'\?$',
                r'^(?:what|how|when|where|why|can|will|should|do)',
            ],
            "Expressing gratitude": [
                r'thank\s+you',
                r'i\s+appreciate',
                r'thanks',
            ],
            "Confirming understanding": [
                r'(?:i\s+)?(?:see|understand|got\s+it)',
                r'(?:that\'?s|it\'?s)\s+(?:a\s+)?relief',
                r'(?:that\'?s|it\'?s)\s+(?:good|great)',
            ]
        }
    
    def detect_intent(self, text: str) -> str:
        """
        Detect the primary intent of patient dialogue
        
        Args:
            text: Patient dialogue text
            
        Returns:
            Intent label
        """
        text_lower = text.lower().strip()
        word_count = len(text.split())
        
        # Score each intent
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            if score > 0:
                intent_scores[intent] = score
        
        # Special validation: "Answering questions" only for short utterances (≤ 8 words)
        # This prevents long symptom descriptions from being misclassified
        if "Answering questions" in intent_scores and word_count > 8:
            # Check if it's a symptom report
            symptom_keywords = ['pain', 'hurt', 'ache', 'cough', 'fever', 'feel', 'discomfort', 'nausea', 'dizzy']
            if any(kw in text_lower for kw in symptom_keywords):
                del intent_scores["Answering questions"]
        
        # Return highest scoring intent
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        
        # Default intent based on keywords
        if any(word in text_lower for word in ['pain', 'hurt', 'ache', 'feel', 'discomfort']):
            return "Reporting symptoms"
        elif any(word in text_lower for word in ['worry', 'concern', 'anxious', 'afraid']):
            return "Seeking reassurance"
        else:
            return "General communication"
    
    def detect_multiple_intents(self, text: str) -> List[str]:
        """
        Detect all applicable intents in text
        
        Args:
            text: Patient dialogue text
            
        Returns:
            List of intent labels
        """
        text_lower = text.lower().strip()
        detected_intents = []
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_intents.append(intent)
                    break
        
        return detected_intents if detected_intents else ["General communication"]


class SentimentIntentAnalyzer:
    """Combined sentiment and intent analysis"""
    
    def __init__(self):
        """Initialize analyzers"""
        self.sentiment_analyzer = SentimentAnalyzer()
        self.intent_detector = IntentDetector()
    
    def analyze(self, text: str) -> Dict:
        """
        Perform combined sentiment and intent analysis
        
        Args:
            text: Patient dialogue text
            
        Returns:
            Dictionary with sentiment and intent
        """
        sentiment = self.sentiment_analyzer.analyze_sentiment(text)
        intent = self.intent_detector.detect_intent(text)
        
        return {
            "Sentiment": sentiment,
            "Intent": intent
        }
    
    def analyze_conversation(self, transcript: str) -> List[Dict]:
        """
        Analyze sentiment and intent for each patient utterance in conversation
        
        Args:
            transcript: Full conversation transcript
            
        Returns:
            List of analysis results for each patient utterance
        """
        results = []
        
        # Extract patient utterances
        patient_utterances = re.findall(
            r'Patient:\s*([^.]+(?:\.[^P])*)',
            transcript,
            re.IGNORECASE | re.MULTILINE
        )
        
        for utterance in patient_utterances:
            utterance = utterance.strip()
            if utterance:
                analysis = self.analyze(utterance)
                analysis['Text'] = utterance[:100] + '...' if len(utterance) > 100 else utterance
                results.append(analysis)
        
        return results


if __name__ == "__main__":
    # Test sentiment and intent analysis
    analyzer = SentimentIntentAnalyzer()
    
    test_cases = [
        "I'm a bit worried about my back pain, but I hope it gets better soon.",
        "I had a car accident. My neck and back hurt a lot for four weeks.",
        "That's a relief! Thank you, doctor.",
        "I'm doing better, but I still have some discomfort now and then.",
        "Yes, I had ten physiotherapy sessions, and now I only have occasional back pain."
    ]
    
    print("Sentiment & Intent Analysis Results:")
    print("=" * 70)
    
    for text in test_cases:
        result = analyzer.analyze(text)
        print(f"\nText: \"{text}\"")
        print(f"Sentiment: {result['Sentiment']}")
        print(f"Intent: {result['Intent']}")
        print("-" * 70)
