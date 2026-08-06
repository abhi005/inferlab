import os

import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer


class InferenceRunner:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

        print(f"loading tokenizer {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir="/tmp", padding_side="left"
        )
        print(f"loaded tokenizer {model_name}")

        print(f"loading model {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="/tmp")
        print(f"loaded model {model_name}")

        process = psutil.Process(os.getpid())
        print(f"RSS Memory: {process.memory_info().rss / (1024**2):.2f} MB")

    def generate(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")
        print("generating")
        outputs = self.model.generate(**inputs, max_new_tokens=256)
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
