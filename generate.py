import sys
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, AutoTokenizer
from utils.prompter import Prompter

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(base_model, load_8bit, lora_weights):
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    return model

def inference(base_model: str, lora_weights: str, load_8bit: bool = True, prompt_template: str = ""):
    assert (base_model), "Please specify a base_model"
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = load_model(base_model, load_8bit, lora_weights)

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit: model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    instruction='কৃত্রিম বুদ্ধিমত্তার ধারণাটি সহজ ভাষায় ব্যাখ্যা কর।'
    input= ''#'বাংলাদেশ একটি তৃতীয় '
    max_new_tokens=1024

    prompt = prompter.generate_prompt(instruction, input)
    encodeds = tokenizer(prompt, return_tensors='pt')
    model_inputs = encodeds.to(device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids)
    return decoded[0]

if __name__ == "__main__":
    base_model = "NousResearch/Meta-Llama-3-8B"
    lora_weights = "lora-alpaca/checkpoint-200"
    out = inference(base_model, lora_weights)
    print(out)

# Output:
# কৃত্রিম বুদ্ধিমত্তা হল একটি কৃত্রিম ব্যক্তি যা মানুষের মতো বুদ্ধিমত্তা এবং আত্মনির্ভর সিদ্ধান্ত নিতে সক্ষম। এটি অনেকটি কর্মক্ষমতা, অনুমান এবং সিদ্ধান্ত নিতে সক্ষম। এটি সাধারণত কম্পিউটার বা রোবটিক্স দ্বারা ব্যবহৃত হয়।<|end_of_text|>
