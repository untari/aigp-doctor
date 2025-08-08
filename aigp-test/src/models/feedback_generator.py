
class ActionableFeedbackGenerator:
    """Generates actionable feedback based on the final diagnosis."""
    def __init__(self):
        self.feedback_map = {
            "Pneumonia": "Recommendation: Consult a doctor for a chest X-ray and a course of antibiotics.",
            "Viral Infection (Common Cold/Flu)": "Recommendation: Rest, drink plenty of fluids, and manage symptoms with over-the-counter medication.",
            "Bacterial Infection": "Recommendation: See a doctor to get a prescription for antibiotics.",
            "Gastroenteritis": "Recommendation: Stay hydrated and avoid solid foods for a while. If symptoms persist, see a doctor.",
            "Migraine Headache": "Recommendation: Rest in a dark, quiet room and consider over-the-counter pain relievers. If headaches are frequent, consult a neurologist.",
            "Default": "Recommendation: Please consult a healthcare professional for a definitive diagnosis and treatment plan."}

    def generate_feedback(self, diagnosis: str) -> str:
        return self.feedback_map.get(diagnosis, self.feedback_map["Default"])
