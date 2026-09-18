#!/usr/bin/env python
"""Fine-tune the DeBERTa-v3 NLP classifier — Phase E training pipeline.

HONESTY CONTRACT
================
- The base pretrained ``microsoft/deberta-v3-base`` is NOT a phishing
  classifier. This script fine-tunes it into one using a REAL labeled
  dataset that the project must supply. Without such a dataset the script
  exits with a clear message; it never fabricates training results.
- The saved artifact carries a ForentisAI fine-tune marker in its
  ``config.json`` (``_forentisai_finetune_marker``). Inference refuses any
  artifact without that marker (base/foreign models), so an untrained
  DeBERTa can never silently produce "predictions".
- Downloading the base model happens at TRAINING time only. Inference
  (``app.ai.nlp.model.load_nlp_model``) is strictly local.

Dataset format (JSON): ``{"rows": [{"text": "...", "label": "suspicious"|
"benign"}, ...]}``. Register the dataset's real-world source and labeling
methodology in ``ai/datasets/README.md`` before training.

Usage (from the ``backend`` directory, venv active):

    python ../ai/training/train_transformer.py --dataset ../ai/datasets/external/labeled.json
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import argparse
import json
import shutil  # noqa: S404 - local file copies only
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ai" / "models" / "nlp"

BASE_MODEL = "microsoft/deberta-v3-base"
FORENTISAI_NLP_MARKER = "forentisai_nlp_finetuned_v1"  # must match app.ai.nlp.model


def _load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) < 10:
        raise SystemExit(
            "Dataset must be JSON with a 'rows' list of {text, label} objects "
            "(at least 10 rows for a meaningful split)."
        )
    cleaned: list[dict] = []
    for index, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        label = row.get("label")
        if not text or label not in {"suspicious", "benign"}:
            raise SystemExit(f"Row {index} is malformed (needs non-empty text + label).")
        cleaned.append({"text": text, "label": label})
    labels = {row["label"] for row in cleaned}
    if len(labels) < 2:
        raise SystemExit("Dataset must contain both 'suspicious' and 'benign' labels.")
    return cleaned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="Path to a labeled JSON dataset ({'rows': [{'text', 'label'}, ...]}).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--base-model", default=BASE_MODEL,
        help="Base transformer to fine-tune (default: microsoft/deberta-v3-base).",
    )
    args = parser.parse_args()

    try:
        import numpy as np
        import torch
        from datasets import Dataset  # type: ignore[import-not-found]
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as error:
        raise SystemExit(
            "Training dependencies are missing (torch/transformers/datasets). "
            "Install them in the backend venv first. Inference does NOT need "
            f"the 'datasets' package. Original error: {error}"
        ) from error

    rows = _load_rows(args.dataset)
    print(f"Loaded {len(rows)} labeled rows from {args.dataset}")

    # Deterministic split (documented, seeded, no silent shuffling surprises).
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(rows))
    validation_size = max(2, int(len(rows) * 0.2))
    validation_indices = set(indices[:validation_size].tolist())

    train_rows = [row for i, row in enumerate(rows) if i not in validation_indices]
    validation_rows = [row for i, row in enumerate(rows) if i in validation_indices]
    print(f"Split: {len(train_rows)} train / {len(validation_rows)} validation")

    label2id = {"benign": 0, "suspicious": 1}
    id2label = {0: "benign", 1: "suspicious"}

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, max_length=args.max_length
        )

    train_dataset = Dataset.from_list(
        [
            {"text": row["text"], "label": label2id[row["label"]]}
            for row in train_rows
        ]
    ).map(_tokenize, batched=True)
    validation_dataset = Dataset.from_list(
        [
            {"text": row["text"], "label": label2id[row["label"]]}
            for row in validation_rows
        ]
    ).map(_tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    training_arguments = TrainingArguments(
        output_dir=str(args.output_dir / "_training"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="epoch",
        seed=args.seed,
        use_cpu=not torch.cuda.is_available(),
        report_to=[],
    )

    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = float((predictions == labels).mean())
        # Reported honestly as validation-split metrics of THIS run.
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=_compute_metrics,
    )

    print("Starting fine-tuning (this downloads the base model at TRAINING time only)...")
    trainer.train()

    metrics = trainer.evaluate()
    print(f"[honesty] validation metrics for this run: {metrics}")

    # Export final artifact with tokenizer + ForentisAI fine-tune marker.
    output_dir = args.output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    config_path = output_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["_forentisai_finetune_marker"] = FORENTISAI_NLP_MARKER
    config["_forentisai_training_metadata"] = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "base_model": args.base_model,
        "epochs": args.epochs,
        "validation_metrics": metrics,
        "notes": (
            "Validation metrics describe THIS run on THIS dataset only. They "
            "are not real-world accuracy claims. See ai/datasets/README.md."
        ),
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.rmtree(output_dir / "_training", ignore_errors=True)
    print(f"Saved fine-tuned NLP artifact to {output_dir}")
    print(
        "Inference will refuse this artifact if the ForentisAI marker is "
        "removed — base/foreign models are never used for prediction."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
