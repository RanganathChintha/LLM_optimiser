from typing import List

import torch
from datasets import DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from data import prepare_mt5_dataset


def train_mt5(
    dataset: DatasetDict,
    model_name: str = "google/mt5-small",
    output_dir: str = "artifacts/mt5",
    epochs: int = 1,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_len: int = 64,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    tokenized_train = prepare_mt5_dataset(dataset["train"], tokenizer, max_len=max_len)
    tokenized_val = prepare_mt5_dataset(dataset["validation"], tokenizer, max_len=max_len)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        predict_with_generate=True,
        logging_steps=10,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    return model, tokenizer


@torch.no_grad()
def predict_mt5(
    model,
    tokenizer,
    src_texts: List[str],
    max_len: int = 64,
    batch_size: int = 8,
    device: str = "cpu",
) -> List[str]:
    model = model.to(device)
    model.eval()

    outputs = []
    for i in range(0, len(src_texts), batch_size):
        batch = src_texts[i : i + batch_size]
        prompts = [f"translate English to Telugu: {x}" for x in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
        gen_ids = model.generate(
            **enc,
            max_length=max_len,
            num_beams=4,
        )
        outputs.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

    return outputs
