"""The local-expert LSTM (LE-LSTM) and its training loop.

Each token of a routing trace is the multi-hot vector x_t of Eq. 1 over the L*E local experts, with
a one at (l,e) when the router sent that token to expert e in layer l. A single-layer LSTM with
hidden dimension 64 reads the sequence and a linear head maps its final state to a logit.

The representation is explicit rather than a learned embedding of expert ids: a local expert is a
distinct unit of computation, and giving each its own input weight is what lets gradient x input
attribution (l3.attribution) assign credit to (l,e) pairs directly.

The input weight matrix is initialized at zero and the recurrent and head weights at their PyTorch
defaults. With random input weights each expert starts with an arbitrary preference that training
barely moves once the loss has been driven down, so the resulting ranking varies from seed to seed.
The recurrent weights and the batch order still depend on the seed, which is why stage 3 trains an
ensemble of members with different seeds and averages their attributions.
"""

import numpy as np
import torch
import torch.nn as nn

HIDDEN = 64
LR = 1e-3
BETAS = (0.9, 0.999)
EPS = 1e-8
BATCH_SIZE = 16
PREDICT_BATCH_SIZE = 64


class LocalExpertLSTM(nn.Module):
    def __init__(self, n_layers, n_experts):
        super().__init__()
        self.n_layers, self.n_experts = n_layers, n_experts
        self.lstm = nn.LSTM(input_size=n_layers * n_experts, hidden_size=HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, 1)
        with torch.no_grad():
            self.lstm.weight_ih_l0.zero_()

    def multi_hot(self, ids):
        """(B, T, L, K) expert ids -> (B, T, L*E) multi-hot, Eq. 1."""
        b, t, n_layers, _ = ids.shape
        x = ids.new_zeros((b, t, n_layers, self.n_experts), dtype=torch.float32)
        x.scatter_(3, ids.long(), 1.0)
        return x.reshape(b, t, n_layers * self.n_experts)

    def forward_from_multi_hot(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, (h_t, _) = self.lstm(packed)
        return self.head(h_t[-1]).squeeze(-1)

    def forward(self, ids, lengths):
        return self.forward_from_multi_hot(self.multi_hot(ids), lengths)


def collate(traces, idx, labels=None):
    """Pads a batch of (T, L, K) traces to the longest, returning ids, true lengths and, given labels, the labels."""
    batch = [traces[i] for i in idx]
    lengths = torch.tensor([t.shape[0] for t in batch])
    n_layers, top_k = batch[0].shape[1], batch[0].shape[2]
    ids = torch.zeros((len(batch), int(lengths.max()), n_layers, top_k), dtype=torch.long)
    for i, t in enumerate(batch):
        ids[i, : t.shape[0]] = torch.as_tensor(t.astype(np.int64))
    if labels is None:
        return ids, lengths
    return ids, lengths, torch.tensor([float(labels[i]) for i in idx])


def train(n_layers, n_experts, traces, labels, idx, seed, device, epochs, epoch_end=None):
    """A new LocalExpertLSTM, initialised from `seed`, trained with binary cross-entropy on the logits and Adam for a
    given number of epochs; the batch order also follows `seed`.

    epoch_end(epoch, mean_train_loss, model), if given, runs after every epoch with the model in eval
    mode; stage 3 uses it to record the loss on its monitoring fold.
    """
    torch.manual_seed(seed)
    model = LocalExpertLSTM(n_layers, n_experts)
    rng = np.random.default_rng(seed)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)
    criterion = nn.BCEWithLogitsLoss()
    idx = np.asarray(idx)
    for epoch in range(epochs):
        order = idx[rng.permutation(len(idx))]
        total = 0.0
        for s in range(0, len(order), BATCH_SIZE):
            ids, lengths, y = collate(traces, order[s:s + BATCH_SIZE], labels)
            opt.zero_grad()
            loss = criterion(model(ids.to(device), lengths), y.to(device))
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        if epoch_end is not None:
            model.eval()
            epoch_end(epoch + 1, total / len(order), model)
            model.train()
    model.eval()
    return model


@torch.inference_mode()
def predict(model, traces, idx, device):
    """The model's logits for the traces in `idx`, in order."""
    out = []
    for s in range(0, len(idx), PREDICT_BATCH_SIZE):
        ids, lengths = collate(traces, idx[s:s + PREDICT_BATCH_SIZE])
        out.append(model(ids.to(device), lengths).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)
