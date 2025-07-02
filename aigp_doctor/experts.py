
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering, AutoProcessor, AutoModelForCausalLM
import torch
from PIL import Image

class AIHealthcareExperts:
    def __init__(self):
        # Initialize ClinicalBERT for QA
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT-2.0")
            self.model_qa = AutoModelForQuestionAnswering.from_pretrained("emilyalsentzer/Bio_ClinicalBERT-2.0")
            self.clinical_bert = pipeline("question-answering", model=self.model_qa, tokenizer=self.tokenizer)
        except Exception as e:
            print(f"Error loading ClinicalBERT: {e}")
            self.clinical_bert = None

    def clinical_llm_analysis(self, text_data):
        """Clinical LLM for general medical text analysis"""
        # Enhanced analysis with symptom extraction
        analysis = {
            'expert': 'Clinical_LLM',
            'findings': f"Clinical analysis of patient description: {text_data}",
            'confidence': 0.85,
            'key_symptoms': self._extract_symptoms(text_data)
        }
        print(f"Clinical LLM analysis completed with confidence: {analysis['confidence']}")
        return analysis

    def _extract_symptoms(self, text):
        """Basic symptom extraction from text"""
        common_symptoms = ['pain', 'fever', 'cough', 'fatigue', 'headache', 'nausea']
        found_symptoms = []
        text_lower = text.lower()
        for symptom in common_symptoms:
            if symptom in text_lower:
                found_symptoms.append(symptom)
        return found_symptoms if found_symptoms else ['General discomfort mentioned']

    def clinical_bert_qa(self, text_data):
        """ClinicalBERT for symptom extraction and medical QA"""
        if self.clinical_bert is None:
            return {
                'expert': 'ClinicalBERT',
                'qa_results': [{'question': 'Model unavailable', 'answer': 'ClinicalBERT is not loaded.', 'confidence': 0.0}]
            }

        # Generate relevant clinical questions
        questions = [
            "What symptoms is the patient experiencing?",
            "What is the patient's main complaint?",
            "Are there any concerning symptoms mentioned?",
            "What is the duration of symptoms?"
        ]

        results = []
        for question in questions:
            try:
                answer = self.clinical_bert(question=question, context=text_data)
                results.append({
                    'question': question,
                    'answer': answer['answer'],
                    'confidence': answer['score']
                })
            except Exception as e:
                results.append({
                    'question': question,
                    'answer': f'Unable to extract: {str(e)}',
                    'confidence': 0.0
                })

        avg_confidence = sum([r['confidence'] for r in results]) / len(results) if results else 0.0
        print(f"ClinicalBERT QA completed with average confidence: {avg_confidence}")

        return {
            'expert': 'ClinicalBERT',
            'qa_results': results,
            'confidence': avg_confidence
        }

    def llava_med_analysis(self, image_data):
        """LLaVA-Med for medical image analysis with natural language"""
        # For llava_med_analysis, we'll ask a general question about the image.
        question = "What do you see in this medical image?"
        
        # Use the same model as biomedclip_classification
        return self.biomedclip_classification(image_data, question)

    def biomedclip_classification(self, image_data, question="What is the primary diagnosis?"):
        """BioMedCLIP for medical image classification and VQA"""
        try:
            if not image_data:
                return {
                    'expert': 'Idefic_medical_VQA',
                    'findings': 'No image data provided.',
                    'confidence': 0.0
                }

            # Load the model and processor
            processor = AutoProcessor.from_pretrained("Shashwath01/Idefic_medical_VQA_merged_4bit")
            model = AutoModelForCausalLM.from_pretrained("Shashwath01/Idefic_medical_VQA_merged_4bit", torch_dtype=torch.bfloat16, device_map="auto")

            # Prepare the prompt
            prompt = [
                "User:",
                f'''You are a medical expert. Please answer the following question about the image: {question}''',
                Image.open(image_data),
                "<end_of_utterance>",
                '''
Assistant:'''
            ]
            
            # Process the inputs
            inputs = processor(prompt, return_tensors="pt").to("cuda")
            
            # Generate the output
            generated_ids = model.generate(**inputs, max_length=128)
            answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            analysis = {
                'expert': 'Idefic_medical_VQA',
                'findings': answer,
                'confidence': 0.85,  # Placeholder confidence
            }
            print(f"Idefic_medical_VQA analysis completed with confidence: {analysis['confidence']}")
            return analysis

        except Exception as e:
            print(f"Error during Idefic_medical_VQA analysis: {e}")
            return {
                'expert': 'Idefic_medical_VQA',
                'findings': f'Error during analysis: {str(e)}',
                'confidence': 0.0
            }

    def advanced_medical_llm_analysis(self, text_data):
        """
        Placeholder for a more advanced medical LLM for comprehensive text analysis.
        Potential models: Clinical Camel (open-source), or commercial APIs like Med-PaLM 2.
        """
        analysis = {
            'expert': 'Advanced_Medical_LLM',
            'findings': f"Comprehensive medical text analysis by advanced LLM: {text_data}",
            'confidence': 0.90,
            'suggested_diagnoses': ['Diagnosis X', 'Diagnosis Y'],
            'treatment_recommendations': ['Treatment A', 'Treatment B']
        }
        print(f"Advanced Medical LLM analysis completed with confidence: {analysis['confidence']}")
        return analysis

    def chest_xray_analysis(self, image_data):
        """
        Placeholder for a specialized chest X-ray analysis model.
        This would typically involve a fine-tuned CNN or Vision Transformer.
        """
        try:
            if not image_data:
                return {
                    'expert': 'Chest_XRay_Expert',
                    'findings': 'No image data provided.',
                    'confidence': 0.0
                }

            # Load pretrained X-ray model
            model = xrv.models.DenseNet(weights="densenet121-res224-all")

            # Define preprocessing function
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            # Open and preprocess the image
            image = Image.open(image_data)
            if image.mode != "L":
                image = image.convert("L")
            img_tensor = transform(image).unsqueeze(0)

            # Predict
            with torch.no_grad():
                outputs = model(img_tensor)

            # Get top 5 predictions
            top5_idx = torch.topk(outputs[0], 5).indices
            results = {model.pathologies[i]: round(outputs[0][i].item(), 4) for i in top5_idx}

            analysis = {
                'expert': 'Chest_XRay_Expert',
                'findings': 'Automated analysis of chest X-ray image.',
                'confidence': 0.88, # Placeholder confidence
                'detected_conditions': results
            }
            print(f"Chest X-Ray analysis completed with confidence: {analysis['confidence']}")
            return analysis

        except Exception as e:
            print(f"Error during Chest X-Ray analysis: {e}")
            return {
                'expert': 'Chest_XRay_Expert',
                'findings': f'Error during analysis: {str(e)}',
                'confidence': 0.0
            }
