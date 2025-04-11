import torch
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ModelChat:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "saved_model/final_model")
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        print("Loading model from:", self.model_path)
        try:
            # Add model existence check
            if not os.path.exists(self.model_path):
                print(f"Error: Model not found at {self.model_path}")
                return False
                
            self.model = GPT2LMHeadModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,  # Use FP16 for inference
                device_map='auto',
                low_cpu_mem_usage=True
            ).eval()
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
            
            # Add padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def generate_response(self, prompt, max_length=100):
        if not self.model or not self.tokenizer:
            print("Model not loaded!")
            return None
            
        try:
            # Prepare input with proper padding and truncation
            encoded = self.tokenizer.encode_plus(
                prompt,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512  # GPT-2's context window
            ).to('cuda')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    encoded['input_ids'],
                    attention_mask=encoded['attention_mask'],
                    max_length=max_length + len(encoded['input_ids'][0]),
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
            
            # Properly decode only the generated part
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_response = full_response[len(prompt):].strip()
            return new_response if new_response else "Sorry, I couldn't generate a meaningful response."
            
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Sorry, an error occurred while generating the response."

def main():
    chat = ModelChat()
    if not chat.load_model():
        print("Failed to load model. Please check if the model exists in the correct location.")
        return
    
    print("\n=== Chat with your trained model ===")
    print("Type 'quit' to exit")
    print("Type 'clear' to clear the conversation")
    print("-" * 40)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                break
            elif user_input.lower() == 'clear':
                print("\n" * 50)
                continue
            elif not user_input:
                continue
            
            response = chat.generate_response(user_input)
            if response:
                print(f"\nBot: {response}")
                
        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Continuing chat...")

if __name__ == "__main__":
    main()