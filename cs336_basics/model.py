import torch
import torch.nn as nn
import math
import numpy as np
import regex as re
from collections import Counter, defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim):
    x_max = x.max(dim=dim, keepdim=True).values
    x = x-x_max

    exp_x = torch.exp(x)

    return exp_x / exp_x.sum(dim=dim, keepdim=True)

class Linear(nn.Module): #Linear是nn.Module 继承的子类
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        std = math.sqrt(2 / (in_features + out_features))
        self.W = nn.Parameter(torch.empty(out_features, in_features, device = device, dtype = dtype))
        nn.init.trunc_normal_(self.W, mean=0, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return x @ self.W.T

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, device=None, dtype=None):
        super().__init__()
        self.W = nn.Parameter(torch.empty(vocab_size, d_model, device=device, dtype=dtype)) #nn.Parameter登记张量
        nn.init.trunc_normal_(self.W, mean=0, std=1, a=-3, b=3)
    
    def forward(self, token_ids):
        return self.W[token_ids]

class Rmsnorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.W = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x):
        in_dtype = x.dtype #记忆原来的数据类型
        x = x.to(torch.float32) #转换成32位浮点数
        norm = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return ((x / norm) * self.W).to(in_dtype)

class Swiglu(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self, x):
        return self.w2(silu(self.w1(x)) * self.w3(x))

class Rope(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        k = torch.arange(0, d_k, 2, device=device) / d_k
        freqs = 1.0 / (theta ** k)
        i = torch.arange(max_seq_len, device=device)
        angles = torch.outer(i, freqs)
        
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
    
    def forward(self, x, token_positions):
        # 获取 token 对应的 cos 和 sin
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        # 如果 x 是 (batch, num_heads, seq, d_k)
        # 而 token_positions 是 (batch, seq)
        # 我们需要给 cos 和 sin 增加一个维度，变成 (batch, 1, seq, d_k/2) 才能广播
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        
        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1)
    scores = scores / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn = softmax(scores, dim=-1)
    return attn @ V

class Multiheadselfattention(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len=None, theta=None, device=None, dtype=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 这里必须是 q_proj, k_proj, v_proj, output_proj
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if max_seq_len is not None and theta is not None:
            self.rope = Rope(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x, token_positions=None):
        batch, seq, d_model = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        K = K.reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)
        V = V.reshape(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

        if self.rope is not None and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq, seq, device=x.device)).bool()
        out = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).reshape(batch, seq, d_model)
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, d_ff=None, device=None, dtype=None):
        super().__init__()
        self.norm1 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.attn = Multiheadselfattention(d_model, num_heads, max_seq_len, theta, device=device, dtype=dtype)
        self.norm2 = Rmsnorm(d_model, device=device, dtype=dtype)
        self.ff = Swiglu(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        batch, seq, d_model = x.shape
        if token_positions is None:
            token_positions = torch.arange(seq, device=x.device)
        y = x + self.attn(self.norm1(x), token_positions)
        z = y + self.ff(self.norm2(y))
        return z

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta, device=None, dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, theta, d_ff, device=device, dtype=dtype)
            for _ in range(num_layers)
        ])
        self.ln_final = Rmsnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
    
    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

def cross_entropy(logits, targets):
    logits_max = logits.max(dim=-1, keepdim=True).values
    logits = logits - logits_max
    logsum_exp = torch.log(torch.exp(logits).sum(dim=-1, keepdim=True))
    log_probs = logits - logsum_exp
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return -target_log_probs.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}        
        super().__init__(params, defaults)
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                
                t = state["t"] + 1
                m = state["m"]
                v = state["v"]

                m.mul_(beta1).add_(grad, alpha=1-beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)
                lr_t = lr*math.sqrt(1-beta2**t)/(1-beta1**t)
                p.data.add_(p.data, alpha=-lr*wd)
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-lr_t)
                state["t"] = t
        return loss

def lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        return t / T_w*alpha_max
    elif t <= T_c:
        cos_val = math.cos((t-T_w)/(T_c-T_w)*math.pi)
        return alpha_min+0.5*(1+cos_val)*(alpha_max-alpha_min)
    else:
        return alpha_min

def gradient_clipping(parameters, max_norm, eps=1e-6):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            total_norm += (p.grad.data**2).sum()
    total_norm = total_norm.sqrt()
    scale = max_norm / (total_norm + eps)
    if total_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(scale)

def get_batch(x, batch_size, context_length, device):
    n = len(x)
    starts = torch.randint(0, n-context_length, (batch_size,))
    inputs = torch.stack([torch.from_numpy(x[s:s+context_length].astype(np.int64)) for s in starts])
    targets = torch.stack([torch.from_numpy(x[s+1:s+context_length+1].astype(np.int64)) for s in starts])
    return inputs.to(device), targets.to(device)

def save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration":iteration,
    }, out)

def load_checkpoint(src, model, optimizer):
    ckpt = torch.load(src)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["iteration"]


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        # bytes -> id 反查表
        self.bytes_to_id = {v: k for k, v in self.vocab.items()}

        # 把特殊 token 加进 vocab
        for tok in self.special_tokens:
            tok_bytes = tok.encode("utf-8")
            if tok_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = tok_bytes
                self.bytes_to_id[tok_bytes] = new_id

        # 合并优先级
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def _encode_word(self, word_bytes):
        """word_bytes: bytes -> list[int]"""
        tokens = [bytes([b]) for b in word_bytes]
        while len(tokens) > 1:
            best_rank = None
            best_i = None
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merge_ranks:
                    r = self.merge_ranks[pair]
                    if best_rank is None or r < best_rank:
                        best_rank = r
                        best_i = i
            if best_rank is None:
                break
            merged = tokens[best_i] + tokens[best_i + 1]
            tokens = tokens[:best_i] + [merged] + tokens[best_i + 2:]
        return [self.bytes_to_id[t] for t in tokens]

    def encode(self, text):
        ids = []
        if self.special_tokens:
            pattern = "(" + "|".join(re.escape(t) for t in self.special_tokens) + ")"
            segments = re.split(pattern, text)
        else:
            segments = [text]

        for seg in segments:
            if seg in self.special_tokens:
                ids.append(self.bytes_to_id[seg.encode("utf-8")])
                continue
            for match in re.finditer(PAT, seg):
                word_bytes = match.group().encode("utf-8")
                ids.extend(self._encode_word(word_bytes))
        return ids

    def encode_iterable(self, iterable):
        for text in iterable:
            for tid in self.encode(text):
                yield tid

    def decode(self, ids):
        byte_seq = b"".join(self.vocab[i] for i in ids)
        return byte_seq.decode("utf-8", errors="replace")

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        import pickle
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

def train_bpe(input_path, vocab_size, special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    # 用集合避免逐个检查 vocab.values()
    vocab_values = set(vocab.values())
    sorted_special = sorted(special_tokens, key=len, reverse=True)
    for tok in sorted_special:
        tok_bytes = tok.encode("utf-8")
        if tok_bytes not in vocab_values:
            vocab[len(vocab)] = tok_bytes
            vocab_values.add(tok_bytes)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if sorted_special:
        pattern = "|".join(re.escape(tok) for tok in sorted_special)
        chunks = re.split(pattern, text)
    else:
        chunks = [text]

    word_freqs = Counter()
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word_bytes = match.group().encode("utf-8")
            word = tuple(bytes([b]) for b in word_bytes)
            word_freqs[word] += 1

    words = [list(word) for word in word_freqs]
    freqs = list(word_freqs.values())
    merges = []

    # pair_freqs 统计 pair 的加权出现次数；
    # pair_to_words 记录包含该 pair 的词，便于只更新受影响的词。
    pair_freqs = Counter()
    pair_to_words = {}

    for word_id, (word, freq) in enumerate(zip(words, freqs)):
        word_pairs = Counter(zip(word, word[1:]))
        for pair, count in word_pairs.items():
            pair_freqs[pair] += count * freq
            pair_to_words.setdefault(pair, set()).add(word_id)

    while len(vocab) < vocab_size and pair_freqs:
        best_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], pair))
        next_token = best_pair[0] + best_pair[1]

        vocab[len(vocab)] = next_token
        merges.append(best_pair)

        # 只有包含 best_pair 的词才可能因本轮合并而改变。
        affected_word_ids = list(pair_to_words.get(best_pair, ()))

        for word_id in affected_word_ids:
            word = words[word_id]
            freq = freqs[word_id]
            old_pairs = Counter(zip(word, word[1:]))

            # 移除旧词对的统计和索引。
            for pair, count in old_pairs.items():
                pair_freqs[pair] -= count * freq
                if pair_freqs[pair] <= 0:
                    del pair_freqs[pair]

            for pair in old_pairs:
                word_ids = pair_to_words[pair]
                word_ids.discard(word_id)
                if not word_ids:
                    del pair_to_words[pair]

            # 按从左到右、不重叠的方式合并。
            merged_word = []
            i = 0
            while i < len(word):
                if (
                    i + 1 < len(word)
                    and word[i] == best_pair[0]
                    and word[i + 1] == best_pair[1]
                ):
                    merged_word.append(next_token)
                    i += 2
                else:
                    merged_word.append(word[i])
                    i += 1

            words[word_id] = merged_word

            # 加入新词对的统计和索引。
            new_pairs = Counter(zip(merged_word, merged_word[1:]))
            for pair, count in new_pairs.items():
                pair_freqs[pair] += count * freq
                pair_to_words.setdefault(pair, set()).add(word_id)

    return vocab, merges