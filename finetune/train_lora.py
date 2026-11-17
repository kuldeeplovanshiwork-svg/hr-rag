"""Brief LoRA fine-tune of Qwen2.5-0.5B-Instruct, sized for an 8 GB RAM laptop (CPU is fine).

  python finetune/make_dataset.py
  python finetune/train_lora.py            # ~10-25 min on a modern laptop CPU

Why LoRA and not QLoRA: QLoRA needs bitsandbytes 4-bit kernels (an NVIDIA GPU). A 0.5B model in
float32 is ~2 GB, so plain LoRA fits in 8 GB RAM without quantisation.
Plain PyTorch loop (no Trainer/TRL) so it keeps working across library versions.
Loss is computed on the answer tokens only (prompt tokens are masked with -100).
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--data", default=str(HERE / "train.jsonl"))
    ap.add_argument("--out", default=str(HERE / "adapter"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()
    model.to(device).train()

    def encode(ex: dict):
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}],
            tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = tok(ex["expected"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p_ids + a_ids)[: args.max_len]
        labels = ([-100] * len(p_ids) + a_ids)[: args.max_len]
        return torch.tensor([ids]), torch.tensor([labels])

    rows = [json.loads(line) for line in open(args.data, encoding="utf-8") if line.strip()]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_total = max(1, (len(rows) * args.epochs) // args.grad_accum)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.05, 1 - s / steps_total))

    losses, step, start = [], 0, time.time()
    for epoch in range(args.epochs):
        random.shuffle(rows)
        running = 0.0
        for i, ex in enumerate(rows, 1):
            ids, labels = encode(ex)
            loss = model(input_ids=ids.to(device), labels=labels.to(device)).loss
            (loss / args.grad_accum).backward()
            running += loss.item()
            if i % args.grad_accum == 0 or i == len(rows):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                step += 1
        avg = running / len(rows)
        losses.append(round(avg, 4))
        print(f"epoch {epoch + 1}/{args.epochs}  mean loss {avg:.4f}  ({time.time() - start:.0f}s)")

    model.save_pretrained(args.out)
    log = {
        "status": "EXECUTED", "base_model": args.model, "train_examples": len(rows), "epochs": args.epochs,
        "lr": args.lr, "lora_rank": args.rank, "grad_accum": args.grad_accum, "seed": args.seed,
        "epoch_mean_loss": losses, "device": device, "seconds": round(time.time() - start, 1),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "peft": peft.__version__},
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "train_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
