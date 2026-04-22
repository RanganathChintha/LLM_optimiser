from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data import SPECIAL_TOKENS, Vocab


class TranslationTensorDataset(Dataset):
    def __init__(self, src_texts: List[str], tgt_texts: List[str], src_vocab: Vocab, tgt_vocab: Vocab):
        self.src = src_texts
        self.tgt = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx: int):
        src_ids = self.src_vocab.encode(self.src[idx], add_sos=False, add_eos=True)
        tgt_ids = self.tgt_vocab.encode(self.tgt[idx], add_sos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, src_pad_id: int, tgt_pad_id: int):
    src_batch, tgt_batch = zip(*batch)
    src_lens = torch.tensor([len(x) for x in src_batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(x) for x in tgt_batch], dtype=torch.long)

    src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=src_pad_id)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_id)

    return src_pad, src_lens, tgt_pad, tgt_lens


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        emb_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, emb_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, emb_dim)

        self.encoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, src_ids: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_emb(src_ids)
        _, (h, c) = self.encoder(src_emb)

        dec_in = tgt_ids[:, :-1]
        dec_emb = self.tgt_emb(dec_in)
        dec_out, _ = self.decoder(dec_emb, (h, c))
        logits = self.fc_out(dec_out)
        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        max_len: int,
        sos_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        src_emb = self.src_emb(src_ids)
        _, (h, c) = self.encoder(src_emb)

        batch_size = src_ids.size(0)
        cur = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=src_ids.device)
        outputs = [cur]

        for _ in range(max_len):
            dec_emb = self.tgt_emb(cur[:, -1:])
            dec_out, (h, c) = self.decoder(dec_emb, (h, c))
            logits = self.fc_out(dec_out[:, -1, :])
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            outputs.append(next_tok)
            cur = torch.cat([cur, next_tok], dim=1)
            if torch.all(next_tok.squeeze(1) == eos_id):
                break

        return torch.cat(outputs, dim=1)


@dataclass
class BaselineArtifacts:
    model: Seq2SeqLSTM
    src_vocab: Vocab
    tgt_vocab: Vocab


def train_baseline(
    train_split,
    val_split,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
) -> BaselineArtifacts:
    model = Seq2SeqLSTM(len(src_vocab.token_to_id), len(tgt_vocab.token_to_id)).to(device)

    train_ds = TranslationTensorDataset(train_split["en"], train_split["te"], src_vocab, tgt_vocab)
    val_ds = TranslationTensorDataset(val_split["en"], val_split["te"], src_vocab, tgt_vocab)

    src_pad_id = src_vocab.token_to_id[SPECIAL_TOKENS["pad"]]
    tgt_pad_id = tgt_vocab.token_to_id[SPECIAL_TOKENS["pad"]]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, src_pad_id, tgt_pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, src_pad_id, tgt_pad_id),
    )

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_id)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for src_ids, _, tgt_ids, _ in tqdm(train_loader, desc=f"Baseline train epoch {epoch}"):
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            logits = model(src_ids, tgt_ids)
            target = tgt_ids[:, 1:]
            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src_ids, _, tgt_ids, _ in val_loader:
                src_ids = src_ids.to(device)
                tgt_ids = tgt_ids.to(device)
                logits = model(src_ids, tgt_ids)
                target = tgt_ids[:, 1:]
                loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
                val_loss += loss.item()

        print(
            f"[Baseline] Epoch {epoch} | train_loss={total_loss / max(len(train_loader), 1):.4f} "
            f"| val_loss={val_loss / max(len(val_loader), 1):.4f}"
        )

    return BaselineArtifacts(model=model, src_vocab=src_vocab, tgt_vocab=tgt_vocab)


@torch.no_grad()
def predict_baseline(
    artifacts: BaselineArtifacts,
    src_texts: List[str],
    max_len: int = 50,
    batch_size: int = 32,
    device: str = "cpu",
) -> List[str]:
    model = artifacts.model.to(device)
    model.eval()

    src_vocab = artifacts.src_vocab
    tgt_vocab = artifacts.tgt_vocab

    src_ids_list = [torch.tensor(src_vocab.encode(x, add_sos=False, add_eos=True), dtype=torch.long) for x in src_texts]

    src_pad_id = src_vocab.token_to_id[SPECIAL_TOKENS["pad"]]
    sos_id = tgt_vocab.token_to_id[SPECIAL_TOKENS["sos"]]
    eos_id = tgt_vocab.token_to_id[SPECIAL_TOKENS["eos"]]

    preds: List[str] = []
    for i in range(0, len(src_ids_list), batch_size):
        batch = src_ids_list[i : i + batch_size]
        src_pad = nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=src_pad_id).to(device)
        out_ids = model.greedy_decode(src_pad, max_len=max_len, sos_id=sos_id, eos_id=eos_id)
        for row in out_ids.cpu().tolist():
            preds.append(tgt_vocab.decode(row))

    return preds
