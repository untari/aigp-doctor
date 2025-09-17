

import torch
from transformers import pipeline

class GeneralAIExpert:
    """Using a general-purpose AI model for conversational answers."""
    def __init__(self):
        print("🔄 Loading General AI Expert (TinyLlama)...")
        try:
            self.text_generator = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
            print("✅ General AI Expert (TinyLlama) loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading General AI Expert: {e}")
            self.text_generator = None

    def answer(self, question: str) -> str:
        if not self.text_generator: return "The General AI Expert is currently unavailable."
        try:
            messages = [{"role": "user", "content": question}]
            prompt = self.text_generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.text_generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
            
            # Clean response processing
            generated_text = outputs[0]["generated_text"]
            
            # Remove the original prompt from the response
            if prompt in generated_text:
                answer = generated_text.replace(prompt, "").strip()
            else:
                # Fallback: try to extract after model token
                if "<start_of_turn>model\n" in generated_text:
                    answer = generated_text.split("<start_of_turn>model\n")[-1].strip()
                else:
                    # Last fallback: remove the user input if it appears at start
                    answer = generated_text
                    if answer.startswith(question):
                        answer = answer[len(question):].strip()
            
            # Remove any remaining chat template artifacts
            answer = answer.replace("<|user|>", "").replace("<|assistant|>", "")
            answer = answer.replace("<start_of_turn>", "").replace("<end_of_turn>", "")
            
            # Clean up any leading/trailing whitespace and newlines
            answer = answer.strip()
            
            # If answer is too short or empty, provide a fallback
            if len(answer) < 10:
                return f"Based on your question about '{question}', I recommend consulting with a healthcare professional for proper medical advice."
            
            return answer
            
        except Exception as e:
            print(f"Error during text generation: {e}")
            return "I encountered an error trying to generate a response."

