import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from .embeddings import AttributeEmbedding



def selective_scan(x: torch.Tensor, dA: torch.Tensor, dB: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    B, L, D = x.shape
    N = dA.shape[-1]
    h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
    ys = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
    
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
        ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        
    return ys

class SelectiveSSM(nn.Module):

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = 0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or max(d_model // 16, 1)

        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        dt_init = torch.exp(
            torch.rand(d_model) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_softplus = torch.log(torch.expm1(dt_init))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_model, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = self.d_state

        x_proj = self.x_proj(x)                        
        dt, B_param, C_param = x_proj.split(
            [self.dt_rank, N, N], dim=-1
        )

        dt = self.dt_proj(dt)                          
        dt = F.softplus(dt)                           

        A = -torch.exp(self.A_log)
        
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)

        y = selective_scan(x, dA, dB, C_param)

        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y


class MambaBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand

        self.norm = nn.LayerNorm(d_model)

        # FiLM layers: predict scale (gamma) and shift (beta) from attribute embedding
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_beta = nn.Linear(d_model, d_model)
        
        # Initialize FiLM to identity (gamma=1, beta=0)
        nn.init.ones_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.ssm = SelectiveSSM(self.d_inner, d_state=d_state)

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attr_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            attr_emb: [B, L, D] or [B, D]
        """
        residual = x
        x = self.norm(x)

        # Apply FiLM modulation
        gamma = self.film_gamma(attr_emb) # [B, L, D]
        beta = self.film_beta(attr_emb)   # [B, L, D]
        x = x * gamma + beta

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :x.shape[1]]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)
        x_branch = self.ssm(x_branch)

        z = F.silu(z)

        out = x_branch * z
        out = self.out_proj(out)
        out = self.dropout(out)

        return residual + out

class ColumnEncoder(nn.Module):

    def __init__(
        self,
        num_tile_types: int = 13,
        column_height: int = 14,
        tile_embed_dim: int = 8,
        d_model: int = 128,
        columns_per_token: int = 1,
    ):
        super().__init__()
        self.num_tile_types = num_tile_types
        self.column_height = column_height
        self.columns_per_token = columns_per_token
        self.tile_embed_dim = tile_embed_dim
        self.d_model = d_model

        self.tile_embedding = nn.Embedding(num_tile_types, tile_embed_dim)
        flat_dim = column_height * columns_per_token * tile_embed_dim

        self.projection = nn.Sequential(
            nn.Linear(flat_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, H_eff] where H_eff = column_height * columns_per_token
        B, L, Heff = tokens.shape
        tile_embs = self.tile_embedding(tokens) # [B, L, Heff, tile_embed_dim]
        flat = tile_embs.reshape(B, L, Heff * self.tile_embed_dim)
        return self.projection(flat)


class ColumnDecoder(nn.Module):
    def __init__(
        self,
        num_tile_types: int = 13,
        column_height: int = 14,
        d_model: int = 128,
        columns_per_token: int = 1,
    ):
        super().__init__()
        self.num_tile_types = num_tile_types
        self.column_height = column_height
        self.columns_per_token = columns_per_token

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, column_height * columns_per_token * num_tile_types),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        out = self.head(x)
        # Reshape to [B, L, K*H, num_tile_types]
        return out.reshape(B, L, self.column_height * self.columns_per_token, self.num_tile_types)


class Mamba(nn.Module):

    def __init__(
        self,
        num_tile_types: int = 13,
        column_height: int = 14,
        tile_embed_dim: int = 8,
        d_model: int = 128,
        n_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        num_attributes: int = 3,
        columns_per_token: int = 1,
        attribute_mappings: Optional[Dict] = None,
    ):
        super().__init__()
        self.num_tile_types = num_tile_types
        self.column_height = column_height
        self.columns_per_token = columns_per_token
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.num_attributes = num_attributes
        
        # Attribute mappings: {attr_idx: [tile_indices]}
        # If None, use defaults for Mario
        if attribute_mappings is None:
            self.attribute_mappings = {
                0: [5],     # Enemies: E only
                1: [2],     # Gaps: - (check bottom)
                2: [8],     # Pipes: [ (one per pipe)
            }
        else:
            self.attribute_mappings = attribute_mappings

        self.column_encoder = ColumnEncoder(
            num_tile_types=num_tile_types,
            column_height=column_height,
            tile_embed_dim=tile_embed_dim,
            d_model=d_model,
            columns_per_token=columns_per_token,
        )

        self.attribute_embedding = AttributeEmbedding(
            num_attributes=num_attributes,
            embedding_dim=d_model,
        )

        self.input_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        self.drop_emb = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        self.column_decoder = ColumnDecoder(
            num_tile_types=num_tile_types,
            column_height=column_height,
            d_model=d_model,
            columns_per_token=columns_per_token,
        )

        # Attribute predictor for auxiliary loss
        self.attribute_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, num_attributes)
        )

        self._initialize_weights()
        self._print_summary()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.8)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        last_linear = self.column_decoder.head[-1]
        nn.init.xavier_uniform_(last_linear.weight, gain=0.1)
        nn.init.zeros_(last_linear.bias)

    def _print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n{'='*70}")
        print(f"Mamba Initialized")
        print(f"{'='*70}")
        print(f"  Tile types:   {self.num_tile_types}")
        print(f"  Column height:{self.column_height}")
        print(f"  Model dim:    {self.d_model}")
        print(f"  Layers:       {self.n_layers}")
        print(f"  Max seq len:  {self.max_seq_len}")
        print(f"  Columns per token:  {self.columns_per_token}")
        print(f"  Parameters:   {total:,}  (trainable: {trainable:,})")
        print(f"{'='*70}\n")

    def forward(
        self,
        column_sequence: torch.Tensor,
        attribute_sequence: torch.Tensor,
    ) -> torch.Tensor:

        B, L, H = column_sequence.shape
        device = column_sequence.device

        col_emb = self.column_encoder(column_sequence)
        attr_emb = self.attribute_embedding(attribute_sequence)

        # Combined initial input
        x = torch.cat([col_emb, attr_emb], dim=-1)
        x = self.input_proj(x)

        x = x + self.pos_embedding[:, :L, :]
        x = self.drop_emb(x)

        for layer in self.layers:
            # Pass attr_emb to each block for FiLM modulation
            x = layer(x, attr_emb)

        x = self.output_norm(x)

        logits = self.column_decoder(x)
        
        # Predict attributes from hidden state for auxiliary loss
        pred_attrs = self.attribute_predictor(x)

        return logits, pred_attrs

    @torch.no_grad()
    def generate(
        self,
        num_columns: int,
        attributes: torch.Tensor,
        initial_column: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        cfg_scale: float = 3.0,
        device: str = "cuda",
    ) -> torch.Tensor:

        self.eval()

        if initial_column is None:
            # Token shape: [1, K * H]
            initial_column = torch.full(
                (1, self.columns_per_token * self.column_height), 2, dtype=torch.long, device=device
            )  
            # Set the bottom tile of the last column in the token to 0 (Ground)
            initial_column[0, -1] = 0
        else:
            initial_column = initial_column.to(device).long()
            if initial_column.dim() == 1:
                initial_column = initial_column.unsqueeze(0)

        # target_attrs shape: [B, K] - the total counts requested
        target_attrs = attributes.to(device).float()
        if target_attrs.dim() == 1:
            target_attrs = target_attrs.unsqueeze(0)
        
        # We'll maintain a list of current "remaining counts" for each step
        # starting with the target
        remaining_counts = target_attrs.clone()

        columns = [initial_column]

        # Target steps: if columns_per_token is 2, we generate half the number of steps
        num_steps = max(1, num_columns // self.columns_per_token)

        # Density scaling: the model was trained on windows of max_seq_len columns.
        # remaining=2 in training means "2 items in ~max_seq_len columns."
        # During generation of num_steps columns, we scale so the model sees
        # the expected count within its learned window size, not the total.
        def _scale_remaining(raw_remaining, steps_left):
            """Scale raw remaining counts to training-window density."""
            if steps_left <= self.max_seq_len:
                return raw_remaining  # No scaling needed for short sequences
            scale = self.max_seq_len / steps_left
            return raw_remaining * scale

        # Initial scaled conditioning
        scaled = _scale_remaining(remaining_counts, num_steps)
        history_counts = [scaled.unsqueeze(1)]

        for i in range(num_steps):
            seq = torch.stack(columns, dim=1)
            cond_seq = torch.cat(history_counts, dim=1)

            if seq.shape[1] > self.max_seq_len:
                seq = seq[:, -self.max_seq_len:]
                cond_seq = cond_seq[:, -self.max_seq_len:]

            cond_logits, _ = self.forward(seq, cond_seq)
            cond_logits = cond_logits[:, -1, :, :]

            if cfg_scale > 1.0:
                uncond_attr = torch.full_like(cond_seq, -1.0)
                uncond_logits, _ = self.forward(seq, uncond_attr)
                uncond_logits = uncond_logits[:, -1, :, :]
                next_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                next_logits = cond_logits

            next_column = self._sample_column(
                next_logits.squeeze(0), temperature, top_k, top_p
            )

            # Update remaining counts
            current_counts = self._count_token_attributes(next_column)
            remaining_counts = (remaining_counts - current_counts).clamp(min=0)

            # Scale for the model's conditioning window
            steps_left = num_steps - (i + 1)
            scaled = _scale_remaining(remaining_counts, max(steps_left, 1))

            columns.append(next_column.unsqueeze(0))
            history_counts.append(scaled.unsqueeze(1))

        # [num_steps, K * H]
        generated_tokens = torch.stack([c.squeeze(0) for c in columns[1:]], dim=0)
        # Reshape back to individual columns [num_columns, H]
        generated = generated_tokens.reshape(-1, self.column_height)
        return generated

    def _count_token_attributes(self, token: torch.Tensor) -> torch.Tensor:
        """Count attributes in a multi-column token."""
        K = self.columns_per_token
        H = self.column_height
        columns = token.reshape(K, H)
        
        total_counts = torch.zeros(self.num_attributes, device=token.device)
        for i in range(K):
            total_counts += self._count_column_attributes(columns[i])
        return total_counts

    def _count_column_attributes(self, column: torch.Tensor) -> torch.Tensor:
        """Count attributes in a single generated column."""
        counts = torch.zeros(self.num_attributes, device=column.device)
        
        for attr_idx, tile_indices in self.attribute_mappings.items():
            if attr_idx == 1:  # Gaps: check bottom row only
                if column[-1] in tile_indices:
                    counts[attr_idx] = 1.0
            else:  # Enemies, Pipes: check any tile in column
                for t_idx in tile_indices:
                    if any(column == t_idx):
                        counts[attr_idx] = 1.0
                        break
        
        return counts

    def _sample_column(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:

        H, C = logits.shape

        if top_k > 0:
            top_k_vals = torch.topk(logits, min(top_k, C), dim=-1)[0]
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            logits = logits.masked_fill(logits < threshold, float('-inf'))

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_mask = cumulative_probs > top_p
            sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
            sorted_mask[:, 0] = False
            indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            logits = logits.masked_fill(indices_to_remove, float('-inf'))

        if temperature > 0.0:
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            sampled = torch.argmax(logits, dim=-1)

        return sampled
