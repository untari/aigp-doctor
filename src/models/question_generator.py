
from typing import List

class DynamicQuestionGenerator:
    """Generates dynamic follow-up questions based on the diagnosis."""
    def __init__(self):
        self.question_bank = {
            "Viral Infection (Common Cold/Flu)": ["How high is the fever?", "Is there a sore throat or cough?"],
            "Bacterial Infection": ["Is the fever consistently high?", "Is there pain in a specific part of the body?"],
            "Pneumonia": ["Is the cough producing any colored phlegm?", "Is there any pain when taking a deep breath?"],
            "Gastroenteritis": ["How long have the symptoms been present?", "Is there any blood in the stool or vomit?"],
            "Migraine Headache": ["Is the headache on one side of the head?", "Are there any visual disturbances, like flashing lights?"],
            "Default": ["Can you provide more details about the symptoms?", "When did the symptoms start?"]}

    def generate_questions(self, diagnosis: str) -> List[str]:
        return self.question_bank.get(diagnosis, self.question_bank["Default"])
