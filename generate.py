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
max_new_tokens = 500
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
    load_in_8bit=True,
    # quantization_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type='nf4',
    # ),
    low_cpu_mem_usage=True
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
#           "3. Calculate average."
#           "4. Print the result."
# ) # codellama-7B 8bit can solve this

# user_question = """The user provided a query that you need to help achieving: Pie plot of 5 smallest prices. Make legend show car models.. 
# You also have a list of subtasks to be accomplished using Python.

# You have been presented with a pandas dataframe named `df`.
# The dataframe df has already been defined and populated with the required data, so don't create a new one.
# The result of `print(df.head(1))` is:
#   Car Model  Engine Size  Fuel Efficiency (L/100km)  Horsepower  Torque  \
# 0   Model_1          1.5                       10.0         228     282   

#    Weight  Top Speed  Acceleration (0-60 mph)  Price  
# 0    2279        315                     6.79  53900  

# Return only the python code that accomplishes the following tasks:
# 1. Sort the DataFrame `df` in ascending order based on the 'Price' column.
# 2. Extract the first 5 rows from the sorted DataFrame.
# 3. Create a pie plot with the 'Price' column as the data and the 'Car Model' column as the labels.
# 4. Set the legend to show the car models.
# 5. Save the pie plot to 'car_specs_Num7.png'.

# Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
# Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
# You must include the neccessery import statements at the top of the code.
# You must include print statements to output the final result of your code.
# You must use the backticks to enclose the code.

# Example of the output format:
# ```python

# ```
# """  # codellama-7B 8bit can solve this!

user_question = """The user provided a query that you need to help achieving: Calculate the correlation between price and car model and subtract 0.3. Make legend show car models.. 
You also have a list of subtasks to be accomplished using Python.

You have been presented with a pandas dataframe named `df`.
The dataframe df has already been defined and populated with the required data, so don't create a new one.
The result of `print(df.head(1))` is:
  Car Model  Engine Size  Fuel Efficiency (L/100km)  Horsepower  Torque  \
0   Model_1          1.5                       10.0         228     282   

   Weight  Top Speed  Acceleration (0-60 mph)  Price  
0    2279        315                     6.79  53900  

Return only the python code that accomplishes the following tasks:
1. Calculate the correlation between 'Price' and 'Car Model' columns.
2. Subtract 0.3 from the found value.
5. Print the result.

Approach each task from the list in isolation, advancing to the next only upon its successful resolution. 
Strictly follow to the prescribed instructions to avoid oversights and ensure an accurate solution.
You must include the neccessery import statements at the top of the code.
You must include print statements to output the final result of your code.
You must use the backticks to enclose the code.

Example of the output format:
```python

```
""" # Works even with my 'oneline' PEFT on...

# user_question = "BeautifulSoup search string 'Elsie' inside tag 'a'"
# user_question = "Select 5 maximal values from column 'GDP' from a pandas dataframe 'df'"

prompt = "<s>[INST] {user_question} [/INST]"

def generate(model, user_question, max_new_tokens=max_new_tokens, top_p=top_p, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')
    # inputs = tokenizer(user_question[0], return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        max_new_tokens=max_new_tokens,
        # **inputs, 
        # generation_config=GenerationConfig(
        #     # do_sample=True,
            # max_new_tokens=max_new_tokens,
        #     # top_p=top_p,
        #     # temperature=temperature,
        # )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model, user_question)
# import pdb; pdb.set_trace()
