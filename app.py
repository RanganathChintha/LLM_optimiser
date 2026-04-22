from fastapi import FastAPI
from pydantic import BaseModel

import torch

from baseline import predict_baseline, train_baseline
from data import build_baseline_vocabs, load_parallel_data
from mt5 import predict_mt5, train_mt5


app = FastAPI(title="EN->TE Translation Compare API")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TranslateRequest(BaseModel):
    text: str


state = {
    "ready": False,
    "baseline_artifacts": None,
    "mt5_model": None,
    "mt5_tokenizer": None,
}


@app.on_event("startup")
def startup():
    dataset = load_parallel_data(train_size=200, val_size=50, test_size=50)
    src_vocab, tgt_vocab = build_baseline_vocabs(dataset["train"])
    state["baseline_artifacts"] = train_baseline(
        dataset["train"], dataset["validation"], src_vocab, tgt_vocab, epochs=1, device=DEVICE
    )
    state["mt5_model"], state["mt5_tokenizer"] = train_mt5(dataset, epochs=1)
    state["ready"] = True


@app.get("/health")
def health():
    return {"ready": state["ready"]}


@app.post("/translate")
def translate(req: TranslateRequest):
    if not state["ready"]:
        return {"error": "Models not ready"}

    baseline_pred = predict_baseline(state["baseline_artifacts"], [req.text], device=DEVICE)[0]
    mt5_pred = predict_mt5(state["mt5_model"], state["mt5_tokenizer"], [req.text], device=DEVICE)[0]

    return {
        "input": req.text,
        "baseline": baseline_pred,
        "mt5": mt5_pred,
    }
