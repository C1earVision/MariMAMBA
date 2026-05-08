import math
import torch
import torch.nn as nn


class AttributeEmbedding(nn.Module):
    def __init__(self, num_attributes: int, embedding_dim: int):
        super().__init__()
        self.num_attributes = num_attributes
        self.embedding_dim = embedding_dim
        

        self.attr_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim // 4),
                nn.SiLU(),
                nn.Linear(embedding_dim // 4, embedding_dim)
            ) for _ in range(num_attributes)
        ])
        
        self.final_projection = nn.Sequential(
            nn.Linear(embedding_dim * num_attributes, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, attributes: torch.Tensor) -> torch.Tensor:
        
        B = attributes.shape[0]

        if attributes.dim() == 2:
            L = 1
            attrs = attributes.unsqueeze(1)
        else:
            L = attributes.shape[1]
            attrs = attributes


        projected = []
        for i in range(self.num_attributes):
            attr_val = attrs[:, :, i:i+1].float()
            projected.append(self.attr_projections[i](attr_val))
            

        combined = torch.cat(projected, dim=-1)
        out = self.final_projection(combined)
        
        if attributes.dim() == 2:
            return out.squeeze(1)
        return out