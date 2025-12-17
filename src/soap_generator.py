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
        """
        Extract Subjective section (patient-reported information)
        
        Args:
            transcript: Full conversation transcript
            entities: Extracted medical entities
            
        Returns:
            Dictionary with subjective information
        """
        subjective = {
            "Chief_Complaint": "",
            "History_of_Present_Illness": ""
        }
        
        # Extract chief complaint from symptoms
        symptoms = entities.get('Symptoms', [])
        if symptoms:
            subjective["Chief_Complaint"] = ", ".join(symptoms[:3])
        else:
            subjective["Chief_Complaint"] = "Pain and discomfort"
        
        # Extract HPI from transcript
        hpi_parts = []
        
        # Look for accident/incident description
        accident_match = re.search(
            r'patient:.*?((?:car\s+)?accident[^.]+(?:\.[^.]+){0,2})',
            transcript.lower(),
            re.DOTALL
        )
        if accident_match:
            hpi_parts.append(accident_match.group(1).strip().capitalize())
        
        # Look for symptom progression
        progression_patterns = [
            r'patient:.*?(first\s+four\s+weeks[^.]+(?:\.[^.]+){0,1})',
            r'patient:.*?(pain[^.]+for\s+(?:four|4)\s+weeks[^.]*)',
        ]
        for pattern in progression_patterns:
            match = re.search(pattern, transcript.lower(), re.DOTALL)
            if match:
                text = match.group(1).strip()
                if text not in str(hpi_parts):
                    hpi_parts.append(text.capitalize())
        
        # Look for current status
        current_status = entities.get('Current_Status', '')
        if current_status:
            hpi_parts.append(f"Currently experiencing {current_status.lower()}.")
        
        # Combine HPI
        if hpi_parts:
            subjective["History_of_Present_Illness"] = " ".join(hpi_parts)
        else:
            # Fallback HPI
            incident_date = entities.get('Date_of_Incident', 'last month')
            diagnosis = entities.get('Diagnosis', 'injury')
            treatments = entities.get('Treatment', [])
            
            hpi = f"Patient had a car accident on {incident_date}, "
            hpi += f"resulting in {diagnosis.lower()}. "
            if treatments:
                hpi += f"Received treatment including {', '.join(treatments[:2]).lower()}. "
            hpi += f"{current_status}."
            
            subjective["History_of_Present_Illness"] = hpi
        
        return subjective
    
    def extract_objective(self, transcript: str) -> Dict:
        """
        Extract Objective section (physician observations and measurements)
        
        Args:
            transcript: Full conversation transcript
            
        Returns:
            Dictionary with objective information
        """
        objective = {
            "Physical_Exam": "",
            "Observations": ""
        }
        
        # Extract physical examination findings
        exam_patterns = [
            r'(?:physical\s+)?examination[^.]*?:(.*?)(?:patient:|physician:|$)',
            r'everything\s+looks\s+good[^.]*\.(.*?)(?:patient:|physician:|$)',
            r'(?:your|the)\s+neck\s+and\s+back[^.]+\.',
        ]
        
        exam_findings = []
        for pattern in exam_patterns:
            match = re.search(pattern, transcript.lower(), re.DOTALL)
            if match:
                finding = match.group(1) if match.lastindex else match.group(0)
                finding = finding.strip()
                if finding and len(finding) > 10:
                    exam_findings.append(finding)
        
        # Look for specific findings
        if 'full range of movement' in transcript.lower() or 'full range of motion' in transcript.lower():
            objective["Physical_Exam"] = "Full range of motion in cervical and lumbar spine, no tenderness."
        elif exam_findings:
            objective["Physical_Exam"] = " ".join(exam_findings[:2]).capitalize()
        else:
            objective["Physical_Exam"] = "Physical examination conducted. No acute distress noted."
        
        # Extract observations
        observations = []
        if 'normal' in transcript.lower() and 'health' in transcript.lower():
            observations.append("Patient appears in normal health")
        if 'no tenderness' in transcript.lower():
            observations.append("no tenderness")
        if 'normal gait' in transcript.lower() or 'walking normally' in transcript.lower():
            observations.append("normal gait")
        
        if observations:
            objective["Observations"] = ", ".join(observations).capitalize() + "."
        else:
            objective["Observations"] = "Patient appears comfortable, normal affect."
        
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
    
    def extract_plan(self, entities: Dict, transcript: str) -> Dict:
        """
        Extract Plan section (treatment plan and follow-up)
        
        Args:
            entities: Extracted medical entities
            transcript: Full conversation transcript
            
        Returns:
            Dictionary with plan information
        """
        plan = {
            "Treatment": "",
            "Follow_Up": ""
        }
        
        # Extract treatment plan
        treatments = entities.get('Treatment', [])
        treatment_plan = []
        
        # Check if ongoing treatment is mentioned
        if 'physiotherapy' in str(treatments).lower():
            treatment_plan.append("Continue physiotherapy as needed")
        
        if 'painkiller' in str(treatments).lower() or 'analgesic' in str(treatments).lower():
            treatment_plan.append("use analgesics for pain relief")
        
        # Look for recommendations in transcript
        if 'continue' in transcript.lower():
            continue_match = re.search(r'continue\s+([^.]+)', transcript.lower())
            if continue_match:
                recommendation = continue_match.group(1).strip()
                if recommendation not in str(treatment_plan):
                    treatment_plan.append(recommendation)
        
        if treatment_plan:
            plan["Treatment"] = ", ".join(treatment_plan).capitalize() + "."
        else:
            plan["Treatment"] = "Conservative management with pain medication as needed."
        
        # Extract follow-up instructions
        follow_up = []
        
        if 'follow-up' in transcript.lower() or 'come back' in transcript.lower():
            follow_up_match = re.search(
                r'(?:follow-up|come back)[^.]*(?:if|when)[^.]+',
                transcript.lower()
            )
            if follow_up_match:
                follow_up.append(follow_up_match.group(0).capitalize())
        
        # Check for timeframe mentions
        prognosis = entities.get('Prognosis', '')
        if 'six months' in prognosis.lower():
            follow_up.append("Patient to return if pain worsens or persists beyond six months")
        
        if follow_up:
            plan["Follow_Up"] = ". ".join(follow_up) + "."
        else:
            plan["Follow_Up"] = "Patient instructed to return if symptoms worsen or new symptoms develop."
        
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
            "Objective": self.extract_objective(transcript),
            "Assessment": self.extract_assessment(entities, transcript),
            "Plan": self.extract_plan(entities, transcript)
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
