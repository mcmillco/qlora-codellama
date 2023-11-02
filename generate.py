import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: 
            print("what")
            return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    print("isdir is false")
    return None, False # first training


# TODO: Update variables
max_new_tokens = 200
top_p = 1
temperature = 1e-9

# Base model
# model_name_or_path = 'huggyllama/llama-7b'
model_name_or_path = "codellama/CodeLlama-7b-Instruct-hf"
# model_name_or_path = "HuggingFaceH4/zephyr-7b-alpha"
# Adapter name on HF hub or local checkpoint path.
adapter_path, _ = get_last_checkpoint('output/codellama-7b-instruct') # '/output' is not accepted
# adapter_path = "output/checkpoint-270"
# adapter_path = 'timdettmers/guanaco-7b'

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# Fixing some of the early LLaMA HF conversion issues.
# tokenizer.bos_token_id = 1
# tokenizer.pad_token = tokenizer.eos_token

print("Loaded tokenizer")

# Load the model (use bf16 for faster inference)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=False,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    , low_cpu_mem_usage=True
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()


# user_question = "What is Einstein's theory of relativity?"

# prompt = (
#     "A chat between a curious human and an artificial intelligence assistant. "
#     "The assistant gives helpful, detailed, and polite answers to the user's questions. \n"
#     "### Human:\n{user_question}\n"
#     "### Assistant:\n"
# )

# user_question = ("Pandas dataframe 'df' is already initialized and pandas is imported as pd.\n"
#           "1. Select 5 maximal values from column 'GDP'.\n"
#           "2. Multiple these values by 2.\n"
#           "3. Print the resulting column."
# )

# user_question = "BeautifulSoup search string 'Elsie' inside tag 'a'"
user_question = "Select 5 maximal values from column 'GDP' from a pandas dataframe 'df'"

prompt = "<s>[INST] {user_question} [/INST]"

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
    # inputs = tokenizer(user_question[0], return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens
        # **inputs, 
        # generation_config=GenerationConfig(
        #     # do_sample=True,
        #     max_new_tokens=max_new_tokens,
        #     # top_p=top_p,
        #     # temperature=temperature,
        # )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model, user_question)
# import pdb; pdb.set_trace()
