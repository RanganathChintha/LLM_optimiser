import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


SPECIAL_TOKENS = {
    "pad": "<pad>",
    "sos": "<sos>",
    "eos": "<eos>",
    "unk": "<unk>",
}


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def build(cls, sentences: List[str], min_freq: int = 1) -> "Vocab":
        freq: Dict[str, int] = {}
        for sent in sentences:
            for tok in sent.split():
                freq[tok] = freq.get(tok, 0) + 1

        tokens = [
            SPECIAL_TOKENS["pad"],
            SPECIAL_TOKENS["sos"],
            SPECIAL_TOKENS["eos"],
            SPECIAL_TOKENS["unk"],
        ]
        for tok, count in sorted(freq.items()):
            if count >= min_freq:
                tokens.append(tok)

        token_to_id = {t: i for i, t in enumerate(tokens)}
        id_to_token = {i: t for t, i in token_to_id.items()}
        return cls(token_to_id=token_to_id, id_to_token=id_to_token)

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = True) -> List[int]:
        ids = []
        if add_sos:
            ids.append(self.token_to_id[SPECIAL_TOKENS["sos"]])
        for tok in text.split():
            ids.append(self.token_to_id.get(tok, self.token_to_id[SPECIAL_TOKENS["unk"]]))
        if add_eos:
            ids.append(self.token_to_id[SPECIAL_TOKENS["eos"]])
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for idx in ids:
            tok = self.id_to_token.get(int(idx), SPECIAL_TOKENS["unk"])
            if tok in {SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["sos"]}:
                continue
            if tok == SPECIAL_TOKENS["eos"]:
                break
            toks.append(tok)
        return " ".join(toks).strip()


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_parallel_data(
    train_size: int = 1000,
    val_size: int = 200,
    test_size: int = 200,
    seed: int = 42,
) -> DatasetDict:
    ds = load_dataset("opus100", "en-te", split="train")
    ds = ds.shuffle(seed=seed)

    total_needed = train_size + val_size + test_size
    ds = ds.select(range(min(total_needed, len(ds))))

    def _flatten(batch):
        en = []
        te = []
        for tr in batch["translation"]:
            en.append(clean_text(tr["en"]))
            te.append(clean_text(tr["te"]))
        return {"en": en, "te": te}

    flat = ds.map(_flatten, batched=True, remove_columns=ds.column_names)

    train = flat.select(range(0, train_size))
    val = flat.select(range(train_size, train_size + val_size))
    test = flat.select(range(train_size + val_size, train_size + val_size + test_size))

    return DatasetDict(train=train, validation=val, test=test)


def prepare_mt5_dataset(dataset: Dataset, tokenizer: AutoTokenizer, max_len: int = 64) -> Dataset:
    def _tokenize(batch):
        inputs = [f"translate English to Telugu: {x}" for x in batch["en"]]
        model_inputs = tokenizer(inputs, max_length=max_len, truncation=True)
        labels = tokenizer(text_target=batch["te"], max_length=max_len, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(_tokenize, batched=True, remove_columns=dataset.column_names)


def build_baseline_vocabs(dataset: Dataset, min_freq: int = 1) -> Tuple[Vocab, Vocab]:
    src_vocab = Vocab.build(dataset["en"], min_freq=min_freq)
    tgt_vocab = Vocab.build(dataset["te"], min_freq=min_freq)
    return src_vocab, tgt_vocab
