
class ActionableFeedbackGenerator:
    """Generates comprehensive medical recommendations including medicines and home remedies."""
    def __init__(self):
        self.comprehensive_recommendations = {
            "Viral Infection (Common Cold/Flu)": {
                "over_counter_medicines": [
                    "Paracetamol/Acetaminophen (500-1000mg every 6-8 hours for fever/pain)",
                    "Ibuprofen (200-400mg every 6-8 hours for fever/inflammation)", 
                    "Lozenges or throat sprays for sore throat",
                    "Decongestant nasal sprays (max 3 days use)"
                ],
                "home_remedies": [
                    "Rest and sleep 8+ hours daily",
                    "Drink plenty of warm fluids (herbal teas, warm water with honey)",
                    "Gargle with warm salt water (1/2 tsp salt in warm water)",
                    "Use a humidifier or breathe steam from hot shower",
                    "Eat chicken soup or warm broths",
                    "Apply warm compress to forehead/sinuses"
                ],
                "when_to_see_doctor": "If fever >101.3°F for >3 days, difficulty breathing, chest pain, or symptoms worsen after 7 days"
            },
            
            "Bacterial Infection": {
                "over_counter_medicines": [
                    "Paracetamol/Acetaminophen for fever and pain relief",
                    "Ibuprofen for inflammation and pain",
                    "Probiotics to maintain gut health during antibiotic treatment"
                ],
                "home_remedies": [
                    "Complete rest and adequate sleep",
                    "Increase fluid intake (water, herbal teas)",
                    "Eat light, nutritious foods",
                    "Apply cold compress for localized pain/swelling"
                ],
                "when_to_see_doctor": "URGENT: See doctor immediately for antibiotic prescription. Bacterial infections require professional medical treatment."
            },
            
            "Pneumonia": {
                "over_counter_medicines": [
                    "Paracetamol for fever (do not exceed recommended dose)",
                    "Cough suppressants only if cough prevents sleep"
                ],
                "home_remedies": [
                    "Complete bed rest",
                    "Drink warm fluids frequently",
                    "Use a cool-mist humidifier",
                    "Sleep with head elevated",
                    "Practice deep breathing exercises when comfortable"
                ],
                "when_to_see_doctor": "IMMEDIATE medical attention required. Pneumonia needs professional diagnosis and treatment with antibiotics/antivirals."
            },
            
            "Gastroenteritis": {
                "over_counter_medicines": [
                    "Oral rehydration salts (ORS) or electrolyte solutions",
                    "Loperamide (Imodium) for diarrhea (only if no fever)",
                    "Paracetamol for pain (avoid NSAIDs like ibuprofen)"
                ],
                "home_remedies": [
                    "Clear fluids: water, herbal teas, clear broths",
                    "BRAT diet: Bananas, Rice, Applesauce, Toast",
                    "Ginger tea for nausea",
                    "Small frequent meals instead of large ones",
                    "Avoid dairy, fatty, and spicy foods",
                    "Probiotics after acute phase"
                ],
                "when_to_see_doctor": "If severe dehydration, blood in stool, high fever, or symptoms persist >3 days"
            },
            
            "Migraine Headache": {
                "over_counter_medicines": [
                    "Ibuprofen (400mg) or Naproxen at first sign",
                    "Paracetamol + Caffeine combination",
                    "Aspirin (900mg) for adults",
                    "Anti-nausea medication if vomiting"
                ],
                "home_remedies": [
                    "Rest in dark, quiet room",
                    "Apply cold compress to forehead/neck",
                    "Gentle head/neck massage",
                    "Stay hydrated with water",
                    "Practice relaxation techniques",
                    "Maintain regular sleep schedule"
                ],
                "when_to_see_doctor": "If headaches become frequent (>4/month), sudden severe headache, or associated with vision changes"
            },
            
            "Arthritis/Joint Inflammation": {
                "over_counter_medicines": [
                    "Ibuprofen (200-400mg) for inflammation",
                    "Naproxen for longer-lasting relief",
                    "Topical anti-inflammatory gels/creams",
                    "Glucosamine/Chondroitin supplements"
                ],
                "home_remedies": [
                    "Apply heat for stiffness, cold for acute pain/swelling",
                    "Gentle stretching and low-impact exercise",
                    "Maintain healthy weight",
                    "Anti-inflammatory foods (turmeric, ginger, fish)",
                    "Epsom salt baths for muscle relaxation"
                ],
                "when_to_see_doctor": "If joint deformity, inability to use joint, or pain interferes with daily activities"
            },
            
            "Allergic Reaction": {
                "over_counter_medicines": [
                    "Antihistamines: Cetirizine, Loratadine, or Diphenhydramine",
                    "Topical hydrocortisone cream for skin reactions",
                    "Calamine lotion for itching",
                    "Keep emergency epinephrine if prescribed"
                ],
                "home_remedies": [
                    "Identify and avoid allergen triggers",
                    "Cool compresses for itchy/inflamed skin",
                    "Oatmeal baths for widespread skin reactions",
                    "Wear loose, breathable clothing",
                    "Keep environment clean and dust-free"
                ],
                "when_to_see_doctor": "EMERGENCY if difficulty breathing, swelling of face/throat, rapid pulse, or severe whole-body reaction"
            },
            
            "Cardiac Evaluation Needed": {
                "over_counter_medicines": [
                    "DO NOT self-medicate for chest pain",
                    "Aspirin only if advised by emergency services",
                    "Keep any prescribed cardiac medications accessible"
                ],
                "home_remedies": [
                    "Stop all physical activity immediately",
                    "Sit or lie down in comfortable position",
                    "Loosen tight clothing",
                    "Practice slow, deep breathing if not in distress",
                    "Stay calm and avoid panic"
                ],
                "when_to_see_doctor": "IMMEDIATE emergency medical attention required. Call emergency services for chest pain."
            },
            
            "Upper Respiratory Infection": {
                "over_counter_medicines": [
                    "Paracetamol/Ibuprofen for pain and fever",
                    "Decongestant nasal sprays (limit to 3 days)",
                    "Cough drops or throat lozenges",
                    "Expectorants to help clear mucus"
                ],
                "home_remedies": [
                    "Warm salt water gargles (3-4 times daily)",
                    "Steam inhalation with eucalyptus oil",
                    "Honey and lemon in warm water",
                    "Increase fluid intake",
                    "Use humidifier or breathe steam",
                    "Elevate head while sleeping"
                ],
                "when_to_see_doctor": "If symptoms worsen after 7 days, high fever, or difficulty swallowing"
            },
            
            "Food Poisoning": {
                "over_counter_medicines": [
                    "Oral rehydration solutions",
                    "Probiotics after acute phase",
                    "Avoid anti-diarrheal medications initially (body needs to clear toxins)"
                ],
                "home_remedies": [
                    "Rest and avoid solid foods initially",
                    "Clear fluids: water, herbal teas, clear broths",
                    "Gradually reintroduce bland foods (BRAT diet)",
                    "Ginger tea for nausea relief",
                    "Avoid dairy, alcohol, and fatty foods",
                    "Maintain electrolyte balance"
                ],
                "when_to_see_doctor": "If severe dehydration, high fever, blood in stool, or symptoms persist >72 hours"
            }
        }
        
        # Default recommendation for unknown conditions
        self.default_recommendation = {
            "over_counter_medicines": [
                "Paracetamol for general pain/fever relief",
                "Consult pharmacist for appropriate over-counter options"
            ],
            "home_remedies": [
                "Adequate rest and sleep",
                "Stay well hydrated",
                "Maintain balanced nutrition",
                "Monitor symptoms carefully"
            ],
            "when_to_see_doctor": "Consult healthcare professional for proper diagnosis and treatment plan"
        }

    def generate_feedback(self, diagnosis: str) -> str:
        """Generate comprehensive medical recommendations"""
        recommendations = self.comprehensive_recommendations.get(diagnosis, self.default_recommendation)
        
        feedback = f"\n🏥 **MEDICAL RECOMMENDATIONS FOR {diagnosis.upper()}:**\n\n"
        
        # Over-the-counter medicines
        feedback += "💊 **Over-the-Counter Medicines:**\n"
        for medicine in recommendations["over_counter_medicines"]:
            feedback += f"• {medicine}\n"
        feedback += "\n"
        
        # Home remedies
        feedback += "🏠 **Home Remedies & Self-Care:**\n"
        for remedy in recommendations["home_remedies"]:
            feedback += f"• {remedy}\n"
        feedback += "\n"
        
        # When to see doctor
        feedback += "⚠️ **When to See a Doctor:**\n"
        feedback += f"• {recommendations['when_to_see_doctor']}\n\n"
        
        # Medical disclaimer
        feedback += "🚨 **IMPORTANT MEDICAL DISCLAIMER:**\n"
        feedback += "• These are general recommendations for educational purposes only\n"
        feedback += "• Always consult qualified healthcare providers before starting any medication\n"
        feedback += "• Seek immediate medical attention for severe or worsening symptoms\n"
        feedback += "• This AI system cannot replace professional medical advice\n"
        
        return feedback
        
    def generate_emergency_recommendations(self, symptoms: str) -> str:
        """Generate emergency recommendations for severe symptoms"""
        emergency_keywords = [
            "chest pain", "difficulty breathing", "severe bleeding", "unconscious", 
            "severe allergic reaction", "poisoning", "severe burns", "head injury"
        ]
        
        symptoms_lower = symptoms.lower()
        if any(keyword in symptoms_lower for keyword in emergency_keywords):
            return ("🚨 **MEDICAL EMERGENCY DETECTED**\n\n"
                   "• Call emergency services immediately (911/999/112)\n"
                   "• Do not delay seeking professional medical care\n"
                   "• Follow emergency dispatcher instructions\n"
                   "• This situation requires immediate professional intervention\n")
        
        return ""
