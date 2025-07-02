class DiagnosticIntegrator:
    def __init__(self):
        pass

    def synthesize_findings(self, expert_outputs):
        """Combine and synthesize findings from multiple AI experts with enhanced logic"""
        symptoms = []
        possible_conditions = []
        expert_confidences = {}
        xai_insights = [] # Placeholder for XAI insights

        for output in expert_outputs:
            expert_name = output.get('expert', 'Unknown')
            confidence = output.get('confidence', 0.0)
            expert_confidences[expert_name] = confidence

            # Extract symptoms from Clinical LLM
            if expert_name == 'Clinical_LLM':
                expert_symptoms = output.get('key_symptoms', [])
                symptoms.extend(expert_symptoms)
                # Add a simple XAI insight for LLM
                xai_insights.append(f"Clinical LLM identified key symptoms with confidence {confidence:.2f}.")

            # Extract symptoms and QA from ClinicalBERT
            elif expert_name == 'ClinicalBERT':
                qa_results = output.get('qa_results', [])
                for qa in qa_results:
                    if qa['confidence'] > 0.3 and 'symptom' in qa['question'].lower():
                        answer = qa['answer'].lower()
                        symptom_keywords = ['pain', 'fever', 'cough', 'fatigue', 'nausea', 'headache']
                        for keyword in symptom_keywords:
                            if keyword in answer and keyword not in symptoms:
                                symptoms.append(keyword)
                # Add a simple XAI insight for ClinicalBERT
                xai_insights.append(f"ClinicalBERT answered questions with average confidence {confidence:.2f}.")

            # Extract conditions from image analysis (LLaVA-Med and BioMedCLIP)
            elif expert_name in ['LLaVA_Med', 'BioMedCLIP']:
                diagnosis = output.get('primary_diagnosis', output.get('findings'))
                if diagnosis and 'placeholder' not in diagnosis.lower():
                    possible_conditions.append(diagnosis)
                # Add a simple XAI insight for image models
                xai_insights.append(f"{expert_name} identified potential conditions with confidence {confidence:.2f}.")

            # Extract conditions from chest X-ray analysis
            elif expert_name == 'Chest_XRay_Expert':
                detected_conditions = output.get('detected_conditions', {})
                for condition, score in detected_conditions.items():
                    if score > 0.5: # Only include conditions with a confidence score greater than 0.5
                        possible_conditions.append(f"{condition} (confidence: {score})")
                xai_insights.append(f"Chest X-Ray Expert identified potential conditions with confidence {confidence:.2f}.")

        # Remove duplicates and clean up symptoms
        symptoms = list(set([s for s in symptoms if s and s != 'General discomfort mentioned']))

        # Enhanced synthesis logic: Prioritize conditions from higher confidence experts or specific types
        # For a real system, this would involve more complex rules or a meta-model
        final_conditions = []
        if possible_conditions:
            # Simple prioritization: if both image models suggest something, give it more weight
            if len(possible_conditions) > 1 and 'LLaVA_Med' in expert_confidences and 'BioMedCLIP' in expert_confidences:
                final_conditions.append(f"Potential conditions from image analysis: {', '.join(possible_conditions)}")
            else:
                final_conditions.extend(possible_conditions)
        
        if not final_conditions and symptoms: # If no specific conditions, but symptoms exist
            final_conditions.append("Further investigation needed based on reported symptoms.")

        synthesis = {
            'symptoms': symptoms if symptoms else ['general discomfort'],
            'possible_conditions': final_conditions if final_conditions else ['No specific conditions identified yet.'],
            'medications': self._generate_medication_advice(symptoms),
            'home_care': self._generate_home_care_advice(symptoms),
            'when_to_see_doctor': self._generate_doctor_advice(symptoms),
            'xai_insights': xai_insights # Include XAI insights in the synthesis
        }

        print(f"Diagnostic integration completed: {len(symptoms)} symptoms identified. XAI insights generated.")
        return synthesis

    def _generate_home_care_advice(self, symptoms):
        """Generate simple home care advice based on symptoms"""
        advice = []
        if 'cough' in symptoms:
            advice.extend(['drink warm liquids', 'use honey for throat'])
        if 'fever' in symptoms:
            advice.extend(['rest and stay hydrated', 'use cool compress'])
        if 'pain' in symptoms:
            advice.extend(['apply heat or cold as needed', 'gentle stretching'])
        if 'fatigue' in symptoms:
            advice.extend(['get plenty of sleep', 'eat nutritious meals'])

        # Default advice if no specific symptoms
        if not advice:
            advice = ['rest well', 'stay hydrated', 'eat healthy foods']
        return advice

    def _generate_doctor_advice(self, symptoms):
        """Generate advice on when to see doctor"""
        urgent_symptoms = ['chest pain', 'shortness of breath', 'severe pain']
        for symptom in symptoms:
            for urgent in urgent_symptoms:
                if urgent in symptom:
                    return ['see doctor immediately if symptoms worsen', 'call emergency services if needed']
        return ['see doctor if symptoms last more than a few days', 'contact doctor for follow-up']

    def _generate_medication_advice(self, symptoms):
        """Generate medication suggestions based on symptoms"""
        medications = []
        # Over-the-counter medications based on symptoms
        if 'cough' in symptoms:
            medications.append('cough drops or cough syrup')
        if 'fever' in symptoms:
            medications.append('paracetamol or ibuprofen for fever')
        if 'pain' in symptoms or 'headache' in symptoms:
            medications.append('paracetamol or ibuprofen for pain')
        if 'nausea' in symptoms:
            medications.append('anti-nausea medication if needed')
        if 'fatigue' in symptoms:
            medications.append('multivitamins to support energy')

        # For respiratory symptoms
        if 'shortness of breath' in ''.join(symptoms) or 'breathing' in ''.join(symptoms):
            medications.append('see doctor immediately - do not self-medicate')

        # For chest pain
        if 'chest pain' in ''.join(symptoms):
            medications.append('seek immediate medical attention - do not self-medicate')

        # Default if no specific medications needed
        if not medications:
            medications = ['no specific medication needed - focus on rest and recovery']
        return medications