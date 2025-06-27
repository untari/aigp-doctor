class GatingMechanism:
    def __init__(self):
        pass

    def route_to_experts(self, processed_data):
        """Determine which AI experts should be activated based on input data and content analysis"""
        routing_plan = {
            'activate_clinical_llm': False,
            'activate_clinical_bert': False,
            'activate_llava_med': False,
            'activate_biomedclip': False,
            'data_for_experts': {}
        }

        text_data = processed_data.get('text_data', '').lower()
        has_audio = processed_data.get('has_audio', False)
        has_image = processed_data.get('has_image', False)

        # Route text data (including transcribed audio) to appropriate text-based experts
        if text_data:
            routing_plan['data_for_experts']['text'] = processed_data['text_data']
            
            # Activate Clinical LLM and ClinicalBERT if text input is substantial
            if len(text_data.split()) > 5: # Simple check for substantial text
                routing_plan['activate_clinical_llm'] = True
                routing_plan['activate_clinical_bert'] = True
                print(f"Routing substantial text to Clinical LLM and ClinicalBERT: {processed_data['text_data']}")
            elif has_audio: # If only audio, still route to text models
                routing_plan['activate_clinical_llm'] = True
                routing_plan['activate_clinical_bert'] = True
                print(f"Routing transcribed audio to Clinical LLM and ClinicalBERT: {processed_data['text_data']}")

        # Route image data to vision experts
        if has_image:
            routing_plan['data_for_experts']['image'] = processed_data['image_data']
            # Always activate vision models if an image is present
            routing_plan['activate_llava_med'] = True
            routing_plan['activate_biomedclip'] = True
            print("Routing image to LLaVA-Med and BioMedCLIP")

        return routing_plan