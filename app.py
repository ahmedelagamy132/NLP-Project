import torch
from peft import PeftModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from flask import Flask, request, jsonify, render_template
import os

# --- Configuration ---
# Ensure this matches the model name used during fine-tuning
BASE_MODEL_ID = "unsloth/Llama-3.2-1B-Instruct"
# Ensure this path points to your saved .pth file
LORA_WEIGHTS_PATH = "LLAMA32_ft_python_code.pth"
# It's recommended to use an environment variable or other secure method for the token.
ACCESS_TOKEN = "####################" # Your Hugging Face Token

# --- Global variables for model and tokenizer ---
MODEL_FT = None
TOKENIZER = None
DEVICE = None

app = Flask(__name__, template_folder='.') # Serve templates from the current directory

def initialize_model():
    global MODEL_FT, TOKENIZER, DEVICE

    print("Initializing model and tokenizer...")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Load tokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=ACCESS_TOKEN)
    TOKENIZER.pad_token = "<|finetune_right_pad_id|>" # As used in your training script
    TOKENIZER.padding_side = "right"                 # As used in your training script

    # BitsAndBytesConfig for loading base model in 4-bit (as in your training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load base model
    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",  # Automatically uses GPU if available, otherwise CPU
        token=ACCESS_TOKEN
    )

    # LoRA config (must match your training configuration)
    peft_config = LoraConfig(
        lora_alpha=16,
        r=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'k_proj', 'v_proj'],
    )

    # Initialize PeftModel
    print("Initializing PeftModel...")
    MODEL_FT = PeftModel(base_model, peft_config)

    # Load the saved LoRA weights
    print(f"Loading LoRA weights from: {LORA_WEIGHTS_PATH}")
    if not os.path.exists(LORA_WEIGHTS_PATH):
        raise FileNotFoundError(f"LoRA weights file not found: {LORA_WEIGHTS_PATH}")

    state_dict = torch.load(LORA_WEIGHTS_PATH, map_location=DEVICE if str(DEVICE) != "auto" else "cpu")

    # Key transformation logic from your script
    # This part is crucial and depends on how the state_dict was saved and how PeftModel expects it.
    # Your script used: new_key = f"base_{key}"
    # This implies your saved keys might be like "model.layers..." and PeftModel needs "base_model.layers..."
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"base_{key}" # As per your original loading script
        new_state_dict[new_key] = value
    
    print("Applying LoRA weights to the base model...")
    MODEL_FT.load_state_dict(new_state_dict, strict=False) # strict=False is important here

    MODEL_FT = MODEL_FT.eval() # Set to evaluation mode
    print("Fine-tuned model loaded successfully and set to evaluation mode.")


# Adapted from your inference script's `generate` function
def generate_text_from_model(model, prompt_text, tokenizer, max_new_tokens, context_size=512, temperature=0.0, top_k=1):
    model_dtype = next(model.parameters()).dtype
    model_device = next(model.parameters()).device

    # Formatting the prompt as expected by the Llama-3.2 Instruct model
    formatted_prompt = (
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt_text}\n\n"
        f"### Response:\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    idx = tokenizer.encode(formatted_prompt)
    idx = torch.tensor(idx, dtype=torch.long, device=model_device).unsqueeze(0)
    num_prompt_tokens = idx.shape[1]

    # EOS tokens for Llama-3.2 from your script
    eos_ids = [128001, 128009] # <|end_of_text|>, <|eot_id|>

    generated_tokens = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # Ensure context window is respected

        with torch.no_grad():
            outputs = model(input_ids=idx_cond, use_cache=True) # use_cache can be True for faster generation
        
        logits = outputs.logits
        logits = logits[:, -1, :] # Get logits for the last token

        if top_k is not None and top_k > 0:
            top_logits, _ = torch.topk(logits, top_k)
            min_val_to_keep = top_logits[:, [-1]]
            logits = torch.where(
                logits < min_val_to_keep,
                torch.tensor(float('-inf'), device=model_device, dtype=model_dtype),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else: # Greedy decoding
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next.item() in eos_ids:
            break
        
        generated_tokens.append(idx_next.item())
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode generated tokens, excluding the prompt
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=False)

    # Clean up potential trailing special tokens not handled by eos_ids logic
    if generated_text.endswith("<|eot_id|>"):
        generated_text = generated_text[:-len("<|eot_id|>")]
    if generated_text.endswith(tokenizer.eos_token): # Generic EOS just in case
         generated_text = generated_text[:-len(tokenizer.eos_token)]
         
    return generated_text.strip()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def handle_generation():
    if not MODEL_FT or not TOKENIZER:
        return jsonify({"error": "Model not initialized properly. Check server logs."}), 500
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        max_new_tokens = int(data.get('max_new_tokens', 512)) # Default to 512, ensure it's an int
        temperature = float(data.get('temperature', 0.0))   # Allow configuring temperature
        top_k = int(data.get('top_k', 1))                     # Allow configuring top_k

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        print(f"Received prompt: {prompt}")
        generated_text = generate_text_from_model(MODEL_FT, prompt, TOKENIZER, max_new_tokens, temperature=temperature, top_k=top_k)
        print(f"Generated response: {generated_text}")
        
        return jsonify({"generated_text": generated_text})

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    try:
        initialize_model() # Load model on start
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000) # Makes it accessible on your network
    except Exception as e:
        print(f"Failed to start the application: {e}")