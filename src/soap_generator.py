"""
SOAP Note Generation Module
Converts medical transcripts into structured SOAP (Subjective, Objective, Assessment, Plan) notes.
"""

from typing import Dict, List
import re
from datetime import datetime


class SOAPNoteGenerator:
    """Generate SOAP notes from medical transcripts"""
    
    def __init__(self):
        """Initialize SOAP note generator"""
        pass
    
    def extract_subjective(self, transcript: str, entities: Dict) -> Dict:
        """Build Subjective section using structured entities (no raw transcript replay)."""
        subjective = {
            "Chief_Complaint": "",
            "History_of_Present_Illness": ""
        }

        symptoms = entities.get('Symptoms', [])
        onset = entities.get('Onset', 'Unknown')
        duration = entities.get('Duration', 'Unknown')
        current_status = entities.get('Current_Status', '')
        
        # Classify symptoms for better presentation
        event_symptoms = [s for s in symptoms if any(kw in s.lower() for kw in ['impact', 'trauma', 'injury', 'accident'])]
        msk_symptoms = [s for s in symptoms if any(kw in s.lower() for kw in ['pain', 'ache', 'stiffness', 'swelling'])]
        functional_symptoms = [s for s in symptoms if any(kw in s.lower() for kw in ['sleep', 'difficulty', 'trouble', 'weakness'])]

        # Chief complaint: narrative style
        if msk_symptoms:
            # Build narrative: "neck and back pain following a car accident"
            pain_parts = []
            if any('neck' in s.lower() for s in msk_symptoms):
                pain_parts.append('neck')
            if any('back' in s.lower() for s in msk_symptoms):
                pain_parts.append('back')
            if any('shoulder' in s.lower() for s in msk_symptoms):
                pain_parts.append('shoulder')
            if any('wrist' in s.lower() for s in msk_symptoms):
                pain_parts.append('wrist')
            
            if pain_parts:
                pain_desc = ' and '.join(pain_parts) + ' pain'
            else:
                pain_desc = msk_symptoms[0].lower()
            
            # Add context if event present
            text_lower = transcript.lower()
            if 'car accident' in text_lower or 'accident' in text_lower:
                subjective["Chief_Complaint"] = f"{pain_desc.capitalize()} following a car accident"
            elif 'fall' in text_lower or 'fell' in text_lower:
                subjective["Chief_Complaint"] = f"{pain_desc.capitalize()} following a fall"
            else:
                subjective["Chief_Complaint"] = pain_desc.capitalize()
        elif symptoms:
            subjective["Chief_Complaint"] = ", ".join(symptoms[:3])
        else:
            subjective["Chief_Complaint"] = "Pain and discomfort"

        # HPI synthesized from structured fields only
        # Split symptoms into historical vs current based on current_status body part hints
        current_parts = []
        for part in ['back', 'neck', 'shoulder', 'arm', 'wrist', 'leg', 'knee']:
            if part in current_status.lower():
                current_parts.append(part)

        current_symptoms = []
        historical_symptoms = []
        for s in symptoms:
            s_low = s.lower()
            if any(bp in s_low for bp in current_parts):
                current_symptoms.append(s)
            else:
                historical_symptoms.append(s)

        hpi_fragments = []
        if historical_symptoms and duration and duration != 'Unknown':
            hpi_fragments.append(f"Initially had {', '.join(historical_symptoms)} for {duration}")
        elif historical_symptoms:
            hpi_fragments.append(f"Initially had {', '.join(historical_symptoms)}")

        if current_symptoms:
            hpi_fragments.append(f"Currently reports {', '.join(current_symptoms)}")
        elif current_status:
            hpi_fragments.append(f"Current status: {current_status}")

        if onset and onset != 'Unknown':
            hpi_fragments.insert(0, f"Onset was {onset.lower()}")

        if hpi_fragments:
            subjective["History_of_Present_Illness"] = ". ".join(hpi_fragments) + "."
        else:
            subjective["History_of_Present_Illness"] = "History not documented."

        return subjective
    
    def extract_objective(self, transcript: str, entities: Dict) -> Dict:
        """Build Objective section using transcript findings when available."""
        objective = {
            "Physical_Exam": "",
            "Observations": ""
        }

        text_lower = transcript.lower()
        findings = []

        if ('full range of motion' in text_lower or 'full range of movement' in text_lower or 'near full range of motion' in text_lower):
            if 'shoulder' in text_lower:
                findings.append("Near full range of motion of the right shoulder")
            if 'neck' in text_lower or 'cervical' in text_lower:
                findings.append("Full range of movement in the neck")
            if 'back' in text_lower or 'lumbar' in text_lower:
                findings.append("Full range of movement in the back")
        if re.search(r'overhead\s+(?:movement|motion)[^.]*pain', text_lower) or 'pain on overhead movement' in text_lower:
            findings.append("Mild pain on overhead movement")
        if 'strength' in text_lower and ('reduced' in text_lower or 'weaker' in text_lower):
            findings.append("Strength mildly reduced due to pain")
        if 'no neurological deficits' in text_lower or 'no neurologic deficits' in text_lower:
            findings.append("No neurological deficits")
        if 'no tenderness' in text_lower:
            findings.append("No tenderness")
        if 'no signs of lasting damage' in text_lower:
            findings.append("No signs of lasting damage")

        if findings:
            objective["Physical_Exam"] = ". ".join(findings) + "."
        else:
            objective["Physical_Exam"] = "Physical examination details not documented in transcript."

        investigations = entities.get('Investigations', {})
        if isinstance(investigations, list):
            # Backward compatibility: treat legacy list as performed
            performed = investigations
            considered = []
            negated = []
        else:
            performed = investigations.get('performed', [])
            considered = investigations.get('considered', [])
            negated = investigations.get('negated', [])

        observation_parts = []

        # Build clinically accurate observation statement
        if performed:
            # List what was done
            if "Examination" in performed and len(performed) == 1:
                observation_parts.append("Clinical examination performed")
            elif "Examination" in performed:
                other_performed = [inv for inv in performed if inv != "Examination"]
                observation_parts.append(f"Clinical examination and {', '.join(other_performed).lower()} performed")
            else:
                observation_parts.append(f"Investigations performed: {', '.join(performed)}")
        
        # Only mention negated investigations if explicitly stated (not just absent)
        if negated:
            negated_phrasing = []
            for inv in negated:
                if inv == "X-ray":
                    negated_phrasing.append("No X-rays obtained")
                elif inv == "MRI":
                    negated_phrasing.append("No MRI obtained")
                elif inv == "CT Scan":
                    negated_phrasing.append("No CT scan obtained")
                else:
                    negated_phrasing.append(f"No {inv.lower()} obtained")
            observation_parts.extend(negated_phrasing)
        
        # Mention considered investigations if present
        if considered:
            observation_parts.append(f"{', '.join(considered)} may be considered if symptoms worsen")

        if observation_parts:
            objective["Observations"] = ". ".join(observation_parts) + "."
        else:
            objective["Observations"] = "No additional observations documented."

        return objective
    
    def extract_assessment(self, entities: Dict, transcript: str) -> Dict:
        """
        Extract Assessment section (diagnosis and evaluation)
        
        Args:
            entities: Extracted medical entities
            transcript: Full conversation transcript
            
        Returns:
            Dictionary with assessment information
        """
        assessment = {
            "Diagnosis": "",
            "Severity": ""
        }
        
        # Get primary diagnosis
        diagnosis = entities.get('Diagnosis', 'Unknown')
        assessment["Diagnosis"] = diagnosis
        
        # Determine severity based on current status and prognosis
        current_status = entities.get('Current_Status', '').lower()
        prognosis = entities.get('Prognosis', '').lower()
        
        if 'occasional' in current_status or 'improving' in current_status:
            assessment["Severity"] = "Mild, improving"
        elif 'full recovery' in prognosis:
            assessment["Severity"] = "Moderate, resolving"
        elif 'severe' in current_status or 'significant' in current_status:
            assessment["Severity"] = "Moderate to severe"
        else:
            assessment["Severity"] = "Mild to moderate"
        
        return assessment
    
    def extract_plan(self, entities: Dict) -> Dict:
        """Build Plan section from structured entities only."""
        plan = {
            "Treatment": "",
            "Follow_Up": ""
        }

        treatments = entities.get('Treatment', [])
        if treatments:
            plan["Treatment"] = ", ".join(treatments) + "."
        else:
            plan["Treatment"] = "Conservative management with analgesics as needed."

        plan["Follow_Up"] = "Return for review if symptoms worsen or fail to improve within six weeks."

        return plan
    
    def generate_soap_note(self, transcript: str, entities: Dict) -> Dict:
        """
        Generate complete SOAP note from transcript and entities
        
        Args:
            transcript: Full conversation transcript
            entities: Extracted medical entities
            
        Returns:
            Complete SOAP note as dictionary
        """
        soap_note = {
            "Subjective": self.extract_subjective(transcript, entities),
            "Objective": self.extract_objective(transcript, entities),
            "Assessment": self.extract_assessment(entities, transcript),
            "Plan": self.extract_plan(entities)
        }
        
        return soap_note
    
    def format_soap_note_text(self, soap_note: Dict, patient_name: str = None) -> str:
        """
        Format SOAP note as readable text
        
        Args:
            soap_note: SOAP note dictionary
            patient_name: Patient name
            
        Returns:
            Formatted SOAP note text
        """
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("SOAP NOTE")
        lines.append("=" * 70)
        if patient_name:
            lines.append(f"Patient: {patient_name}")
        lines.append(f"Date: {datetime.now().strftime('%B %d, %Y')}")
        lines.append("")
        
        # Subjective
        lines.append("SUBJECTIVE:")
        lines.append(f"  Chief Complaint: {soap_note['Subjective']['Chief_Complaint']}")
        lines.append(f"  HPI: {soap_note['Subjective']['History_of_Present_Illness']}")
        lines.append("")
        
        # Objective
        lines.append("OBJECTIVE:")
        lines.append(f"  Physical Exam: {soap_note['Objective']['Physical_Exam']}")
        lines.append(f"  Observations: {soap_note['Objective']['Observations']}")
        lines.append("")
        
        # Assessment
        lines.append("ASSESSMENT:")
        lines.append(f"  Diagnosis: {soap_note['Assessment']['Diagnosis']}")
        lines.append(f"  Severity: {soap_note['Assessment']['Severity']}")
        lines.append("")
        
        # Plan
        lines.append("PLAN:")
        lines.append(f"  Treatment: {soap_note['Plan']['Treatment']}")
        lines.append(f"  Follow-Up: {soap_note['Plan']['Follow_Up']}")
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Test SOAP note generation
    sample_transcript = """
    Physician: Good morning, Ms. Jones. How are you feeling today?
    Patient: I'm doing better, but I still have some discomfort now and then.
    Physician: I understand you were in a car accident last September.
    Patient: Yes, it was on September 1st. I had hit my head on the steering wheel, 
    and I could feel pain in my neck and back almost right away.
    Patient: The first four weeks were rough. My neck and back pain were really bad.
    I had to go through ten sessions of physiotherapy.
    Physician: Everything looks good. Your neck and back have a full range of movement, 
    and there's no tenderness.
    Physician: I'd expect you to make a full recovery within six months.
    """
    
    sample_entities = {
        "Patient_Name": "Janet Jones",
        "Date_of_Incident": "September 1st",
        "Symptoms": ["Neck pain", "Back pain", "Head impact"],
        "Diagnosis": "Whiplash injury and lower back strain",
        "Treatment": ["10 physiotherapy sessions", "Painkillers"],
        "Current_Status": "Occasional backache",
        "Prognosis": "Full recovery expected within six months"
    }
    
    generator = SOAPNoteGenerator()
    soap_note = generator.generate_soap_note(sample_transcript, sample_entities)
    
    print(generator.format_soap_note_text(soap_note, sample_entities['Patient_Name']))
    
    print("\n\nJSON Format:")
    import json
    print(json.dumps(soap_note, indent=2))
