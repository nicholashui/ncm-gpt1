"""Minimal character-level tokenizer for CPU smoke tests (no network, no HF)."""

from tokenization.tokenizer import Tokenizer


# Printable ASCII plus a few whitespace characters. Specials occupy ids 0-2.
_CHARS = [chr(i) for i in range(32, 127)] + ["\n", "\t"]


class CharTokenizer(Tokenizer):
    """Fixed ASCII char vocab with EOS / PAD / UNK. Self-contained for smoke runs."""

    def __init__(self, pad_vocab_to=128, **kwargs):
        _ = kwargs
        super().__init__()
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|pad|>"
        self.unk_token = "<|unk|>"
        self.itos = [self.eos_token, self.pad_token, self.unk_token] + list(_CHARS)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.eos_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2
        # Embedding table may be larger than the used vocab (matches GPT-1 / OWT padding).
        self.used_vocab_size = len(self.itos)
        self.vocab_size = max(int(pad_vocab_to), self.used_vocab_size)

    @classmethod
    def from_config(cls, config):
        pad = int(getattr(config, "pad_vocab_to", 128))
        return cls(pad_vocab_to=pad)

    def encode(self, text: str):
        unk = self.unk_token_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        specials = {self.eos_token_id, self.pad_token_id, self.unk_token_id}
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in specials:
                continue
            if 0 <= i < self.used_vocab_size:
                tok = self.itos[i]
                if skip_special_tokens and tok in (self.eos_token, self.pad_token, self.unk_token):
                    continue
                out.append(tok)
        return "".join(out)
