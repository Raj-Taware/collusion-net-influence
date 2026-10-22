"""Fine-tune Qwen2.5-7B-Instruct with LoRA to collude covertly, then serve it for the RPS games (Modal).

    python experiments/covert_data.py --games 100 --out data/covert_train.jsonl
    modal run cloud/covert_modal.py                 # train; adapter saved to the "netinf-covert" volume
    modal deploy cloud/covert_modal.py              # serve base model + LoRA "covert" (OpenAI-compatible)
    NETINF_BASE_URL=https://<workspace>--netinf-covert-serve.modal.run/v1 NETINF_MODEL=covert \
        python experiments/llm_rps.py --condition channel --runs 10 --rounds 100 --workers 8
    modal app stop netinf-covert                    # stop paying for the GPU
"""

import json
import subprocess
from pathlib import Path

import modal

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "/vol/adapter"
app = modal.App("netinf-covert")
vol = modal.Volume.from_name("netinf-covert", create_if_missing=True)

train_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0", "transformers==4.46.3", "peft==0.13.2", "accelerate==1.1.1")
vllm_image = modal.Image.debian_slim(python_version="3.11").pip_install("vllm==0.6.3.post1")


@app.function(image=train_image, gpu="A10G", volumes={"/vol": vol}, timeout=2 * 3600)
def train(rows: list[dict], epochs: float = 1.0):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                              TrainingArguments)

    tok = AutoTokenizer.from_pretrained(BASE)

    def encode(r):  # loss only on the completion (+ end-of-turn token)
        msgs = [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}]
        prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        answer = tok(r["completion"], add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")]
        return {"input_ids": prompt + answer, "attention_mask": [1] * (len(prompt) + len(answer)),
                "labels": [-100] * len(prompt) + answer}

    data = [encode(r) for r in rows]
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    Trainer(
        model=model,
        args=TrainingArguments(output_dir="/tmp/out", per_device_train_batch_size=4, gradient_accumulation_steps=4,
                               num_train_epochs=epochs, learning_rate=2e-4, lr_scheduler_type="cosine",
                               warmup_ratio=0.03, bf16=True, logging_steps=20, save_strategy="no", report_to=[],
                               group_by_length=True, seed=0),
        train_dataset=data,
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    ).train()
    model.save_pretrained(ADAPTER)
    tok.save_pretrained(ADAPTER)
    vol.commit()
    print("saved adapter to", ADAPTER)


@app.local_entrypoint()
def main(data: str = "data/covert_train.jsonl", epochs: float = 1.0):
    rows = [json.loads(line) for line in Path(data).read_text().splitlines()]
    print(f"training on {len(rows)} examples")
    train.remote(rows, epochs)


@app.function(image=vllm_image, gpu="A10G", volumes={"/vol": vol}, timeout=3 * 3600, scaledown_window=300)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=900)
def serve():
    # Serves the base model as "Qwen/Qwen2.5-7B-Instruct" (used as the judge) and the adapter as "covert".
    subprocess.Popen(
        f"python -m vllm.entrypoints.openai.api_server --model {BASE} --enable-lora --lora-modules covert={ADAPTER} "
        "--max-lora-rank 16 --max-model-len 4096 --host 0.0.0.0 --port 8000", shell=True)
