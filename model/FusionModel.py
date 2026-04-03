from model.SNP_Model import SNPClassifier 
from model.GM_Model import GM_CNN
import torch
import torch.nn as nn



class FusionModel(torch.nn.Module):
    def __init__(self,
                snp_input_dim = 2098,
                gm_input_dim = 93,
                output_dim = 4,
                attn_dim = 256,
                num_heads = 8
                ):
        super().__init__()

        self.snp_encoder = SNPClassifier(
            input_dim = snp_input_dim,
            gm_input_dim = gm_input_dim,
            output_dim = attn_dim,
            output_dim = output_dim,
            num_heads=num_heads
        )

        self.gm_encoder = GM_CNN(
            input_dim = gm_input_dim,
            output_dim = attn_dim,
            num_heads=num_heads
        )

        self.cross_attn_snp_to_gm = torch.nn.MultiheadAttention(
            attn_dim,
            num_heads,
            dropout=0.3,
            batch_first=True
        )

        self.cross_attn_gm_to_snp = torch.nn.MultiheadAttention(
            attn_dim,
            num_heads,
            dropout=0.3,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(attn_dim)
        self.norm2 = nn.LayerNorm(attn_dim)

        self.fusion = nn.Sequential(
            nn.Linear(attn_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(64, output_dim)
        )

    def forward(self, snp_data, gm_data):
        snp_