import gradio as gr
from input_processor import InputProcessor
from gating import GatingMechanism
from experts import AIHealthcareExperts
from integrator import DiagnosticIntegrator

class OutputGenerator:
    def __init__(self):
        pass

    def generate_final_output(self, synthesis):
        """Generate simple, patient-friendly output"""
        output = {
            'symptoms': ', '.join(synthesis['symptoms']),
            'medications': ', '.join(synthesis['medications']),
            'home_care': ', '.join(synthesis['home_care']),
            'doctor_advice': ', '.join(synthesis['when_to_see_doctor']),
            'formatted_report': self._format_simple_report(synthesis),
            'xai_insights': "\n".join(synthesis['xai_insights']) # Include XAI insights
        }
        return output

    def _format_simple_report(self, synthesis):
        """Format a simple, patient-friendly report"""
        report = "=== Your Health Summary ===\n\n"
        report += f"Symptoms you mentioned: {', '.join(synthesis['symptoms'])}\n\n"
        report += "Medications you can try:\n"
        for med in synthesis['medications']:
            report += f"- {med}\n"
        report += "\n"
        report += "What you can do at home:\n"
        for advice in synthesis['home_care']:
            report += f"- {advice}\n"
        report += "\n"
        report += "When to see a doctor:\n"
        for advice in synthesis['when_to_see_doctor']:
            report += f"- {advice}\n"

        if synthesis['possible_conditions']:
            report += f"\nPossible conditions to discuss with doctor: {', '.join(synthesis['possible_conditions'])}"

        report += "\n\nImportant: This is AI advice only. Always talk to a real doctor."
        
        if synthesis['xai_insights']:
            report += "\n\n=== AI Reasoning Insights ===\n"
            report += "\n".join(synthesis['xai_insights'])

        return report

class AIGPDoctorSystem:
    def __init__(self):
        self.input_processor = InputProcessor()
        self.gating_mechanism = GatingMechanism()
        self.ai_experts = AIHealthcareExperts()
        self.diagnostic_integrator = DiagnosticIntegrator()
        self.output_generator = OutputGenerator()

    def process_patient_case(self, audio_file, text_input, image_file):
        """Main processing pipeline following the architectural flow"""
        print("=== Starting AI GP Doctor Analysis ===")

        # Step 1: Input Processing
        processed_data = self.input_processor.process_inputs(audio_file, text_input, image_file)

        # Step 2: Gating/Routing
        routing_plan = self.gating_mechanism.route_to_experts(processed_data)

        # Step 3: Expert Analysis
        expert_outputs = []
        if routing_plan['activate_clinical_llm']:
            output = self.ai_experts.clinical_llm_analysis(routing_plan['data_for_experts']['text'])
            expert_outputs.append(output)

        if routing_plan['activate_clinical_bert']:
            output = self.ai_experts.clinical_bert_qa(routing_plan['data_for_experts']['text'])
            expert_outputs.append(output)

        if routing_plan['activate_llava_med'] and 'image' in routing_plan['data_for_experts']:
            output = self.ai_experts.llava_med_analysis(routing_plan['data_for_experts']['image'])
            expert_outputs.append(output)

        if routing_plan['activate_biomedclip'] and 'image' in routing_plan['data_for_experts']:
            output = self.ai_experts.biomedclip_classification(routing_plan['data_for_experts']['image'])
            expert_outputs.append(output)

        # Step 4: Diagnostic Integration
        synthesis = self.diagnostic_integrator.synthesize_findings(expert_outputs)

        # Step 5: Output Generation
        final_output = self.output_generator.generate_final_output(synthesis)

        print("=== AI GP Doctor Analysis Complete ===")
        return final_output

def create_gradio_interface():
    ai_system = AIGPDoctorSystem()

    def process_interface(audio_file, text_input, image_file):
        """Interface function for Gradio"""
        try:
            result = ai_system.process_patient_case(audio_file, text_input, image_file)
            # Return simple, patient-friendly outputs
            return (
                result['formatted_report'],
                result['symptoms'],
                result['medications'],
                result['doctor_advice'],
                result['xai_insights'] # New output for XAI insights
            )
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            return (error_msg, error_msg, error_msg, error_msg, error_msg) # Return error for new output as well

    demo = gr.Interface(
        fn=process_interface,
        inputs=[
            gr.Audio(type="filepath", label="Voice/Audio Input (will be converted to text)"),
            gr.Textbox(label="Textual Description", placeholder="Describe symptoms..."),
            gr.Image(type="filepath", label="Medical Images")
        ],
        outputs=[
            gr.Textbox(label="Health Summary", lines=12),
            gr.Textbox(label="Your Symptoms", lines=2),
            gr.Textbox(label="Recommended Medications", lines=3),
            gr.Textbox(label="Doctor Advice", lines=2),
            gr.Textbox(label="AI Reasoning Insights", lines=5) # New Textbox for XAI insights
        ],
        title="AI GP Doctor - Simple Medical Advice",
        description="Tell me your symptoms (by voice or text) and I'll give you some simple advice.",
        examples=[
            [None, "I have chest pain and can't breathe well", None],
            [None, "I have a cough and feel tired", None],
        ]
    )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()