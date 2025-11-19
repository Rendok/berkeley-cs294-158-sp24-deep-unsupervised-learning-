import torch
from torch import nn


class GPT2(nn.Module):
        def __init__(self, vocab_len: int, embedding_dim: int, num_head: int, num_blocks: int, seq_len: int, dropout: float):
            super(GPT2, self).__init__()

            self.embedding_dim = embedding_dim
            self.num_head = num_head
            self.num_blocks = num_blocks

            self.embeddings = nn.Embedding(vocab_len, embedding_dim)
            self.pos_embedding = nn.Embedding(seq_len, embedding_dim)

            self.dropout = nn.Dropout(dropout)

            def make_mlp():
                return nn.Sequential(nn.Linear(embedding_dim, 4 * embedding_dim),
                                     nn.GELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(4 * embedding_dim, embedding_dim))

            self.multihead_attention = nn.ModuleList(
                [nn.MultiheadAttention(embedding_dim, num_head, batch_first=True, dropout=dropout) for _ in range(num_blocks)])
            self.mlp = nn.ModuleList([make_mlp() for _ in range(num_blocks)])
            self.norms1 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_blocks)])
            self.norms2 = nn.ModuleList([nn.LayerNorm(embedding_dim) for _ in range(num_blocks)])
            self.output_layer = nn.Sequential(nn.LayerNorm(embedding_dim),
                                              nn.Linear(embedding_dim, vocab_len, bias=False))

        def forward(self, X: torch.Tensor, attn_mask: torch.Tensor, padding_mask: torch.Tensor):
            E = self.embeddings(X)

            positions = torch.arange(0, X.shape[1], device=X.device)
            h = E + self.pos_embedding(positions)
            h = self.dropout(h)

            for i in range(self.num_blocks):
                h = self.norms1[i](h)
                a1, _ = self.multihead_attention[i](h, h, h, attn_mask=attn_mask, key_padding_mask=padding_mask, need_weights=False, is_causal=True)
                h = h + a1
                h = self.dropout(h)

                h = self.norms2[i](h)
                a2 = self.mlp[i](h)
                h = h + a2
                h = self.dropout(h)

            return self.output_layer(h)