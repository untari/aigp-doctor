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
            
            # Prioritize advanced LLM for comprehensive text analysis if text is substantial
            if len(text_data.split()) > 10: # More substantial text for advanced analysis
                routing_plan['activate_advanced_medical_llm'] = True
                print(f"Routing substantial text to Advanced Medical LLM: {processed_data['text_data']}")
            
            # Fallback to ClinicalBERT for QA if advanced LLM not activated or for specific QA needs
            if not routing_plan.get('activate_advanced_medical_llm') or "question" in text_data.lower():
                routing_plan['activate_clinical_bert'] = True
                print(f"Routing text to ClinicalBERT for QA: {processed_data['text_data']}")

            # Always activate Clinical LLM for general analysis if text is present
            routing_plan['activate_clinical_llm'] = True
            print(f"Routing text to Clinical LLM for general analysis: {processed_data['text_data']}")

        # Route image data to vision experts
        if has_image:
            routing_plan['data_for_experts']['image'] = processed_data['image_data']
            image_type = processed_data.get('image_type', '').lower() # Assuming image_type can be passed in processed_data
            text_data = processed_data.get('text_data', '').lower()

            # Route to specialized chest X-ray expert if image type is specified or text contains keywords
            if "x-ray" in image_type or "xray" in image_type or "chest" in image_type or \
               "x-ray" in text_data or "xray" in text_data or "chest" in text_data:
                routing_plan['activate_chest_xray_expert'] = True
                print("Routing image to Chest X-Ray Expert")
            else:
                # Fallback to general vision models if no specific image type or for other image types
                routing_plan['activate_llava_med'] = True
                routing_plan['activate_biomedclip'] = True
                print("Routing image to LLaVA-Med and BioMedCLIP")

        return routing_plan

        return routing_plan