"""LoRA decision model with the unchanged v4 prompt, EOS head and group softmax.

Loader/trainer helpers for the Qwen3-4B LoRA scaling line (see docs/RESULTS.md
section 10). Requires the optional `peft` package.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))

import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from train import StudentModel, collate, encode_row, sha256_file  # noqa: E402


def create(base, rank=32, alpha=64):
    backbone = AutoModel.from_pretrained(base, torch_dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    backbone = get_peft_model(backbone, cfg)
    model = StudentModel(backbone=backbone).to("cuda")
    return model, AutoTokenizer.from_pretrained(base)


def load(checkpoint, trainable=False):
    checkpoint = Path(checkpoint)
    cfg = json.loads((checkpoint / "experiment.json").read_text())
    backbone = AutoModel.from_pretrained(cfg["base_model"], torch_dtype=torch.bfloat16,
                                         attn_implementation="sdpa")
    backbone = PeftModel.from_pretrained(backbone, checkpoint / "adapter", is_trainable=trainable)
    model = StudentModel(backbone=backbone).to("cuda")
    state = torch.load(checkpoint / "head.pt", weights_only=True, map_location="cpu")
    model.norm.load_state_dict(state["norm"])
    model.head.load_state_dict(state["head"])
    model.train(trainable)
    tok = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    return model, tok


def path_chunks(paths, budget):
    chunk, width = [], 0
    for i, path in enumerate(paths):
        new_width = max(width, len(path))
        if chunk and new_width * (len(chunk) + 1) > budget:
            yield chunk
            chunk, width = [], 0
        chunk.append(i)
        width = max(width, len(path))
    if chunk:
        yield chunk


def forward_paths(model, tok, paths, budget=4096):
    # Each path is independent. Concatenate logits BEFORE applying group softmax.
    pieces = []
    for chunk in path_chunks(paths, budget):
        ids, mask, lengths = collate([paths[i] for i in chunk], tok.pad_token_id, "cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pieces.append(model(ids, mask, lengths))
    return torch.cat(pieces)


@torch.inference_mode()
def predict(model, tok, rows, budget=4096, batch_questions=8):
    preds = {}
    for start in range(0, len(rows), batch_questions):
        batch = rows[start:start + batch_questions]
        paths, sizes = [], []
        for row in batch:
            _, _, rp = encode_row(tok, row, 8192)
            paths.extend(rp)
            sizes.append(len(rp))
        scores = forward_paths(model, tok, paths, budget)
        offset = 0
        for row, size in zip(batch, sizes):
            preds[row["id"]] = scores[offset:offset + size].softmax(-1).cpu()
            offset += size
    return preds


def save(model, tok, optimizer, scheduler, step, config, target, rng_state):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=False)
    model.backbone.save_pretrained(target / "adapter", safe_serialization=True)
    tok.save_pretrained(target / "tokenizer")
    torch.save({"norm": model.norm.state_dict(), "head": model.head.state_dict()}, target / "head.pt")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "step": step, "rng": rng_state}, target / "trainer.pt")
    (target / "experiment.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {str(p.relative_to(target)): sha256_file(p) for p in sorted(target.rglob("*"))
                if p.is_file()}
    (target / "manifest.json").write_text(json.dumps({"step": step, "sha256": manifest}, indent=2) + "\n")
