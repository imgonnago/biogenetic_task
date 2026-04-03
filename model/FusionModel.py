import torch
import torch.nn as nn

class FusionModel(nn.Module):
    """
    SNP와 GM 데이터를 양방향 크로스 어텐션으로 융합하는 모델
    """
    def __init__(self,
                snp_input_dim=2098,
                gm_input_dim=93,
                output_dim=4,
                chunk_size=100,
                cnn_channels=128,
                attn_dim=256,
                num_heads=8):
        super().__init__()
        
        # ====== SNP 인코더 (SNPClassifier 구조를 재사용) ======
        self.snp_chunk_size = chunk_size
        self.snp_num_chunks = (snp_input_dim + chunk_size - 1) // chunk_size
        self.snp_input_dim = snp_input_dim
        
        self.snp_mlp = nn.Sequential(
            nn.Linear(chunk_size, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, attn_dim)
        )
        
        self.snp_norm = nn.LayerNorm(attn_dim)
        self.snp_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=0.3, batch_first=True)
        
        # ====== GM 인코더 (GM_CNN 구조를 재사용) ======
        self.gm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.gm_projection = nn.Linear(cnn_channels, attn_dim)
        self.gm_norm = nn.LayerNorm(attn_dim)
        self.gm_attn = nn.MultiheadAttention(attn_dim, num_heads, dropout=0.3, batch_first=True)
        
        # ====== 양방향 크로스 어텐션 ======
        self.cross_attn_snp_to_gm = nn.MultiheadAttention(attn_dim, num_heads, dropout=0.3, batch_first=True)
        self.cross_attn_gm_to_snp = nn.MultiheadAttention(attn_dim, num_heads, dropout=0.3, batch_first=True)
        
        self.cross_norm1 = nn.LayerNorm(attn_dim)
        self.cross_norm2 = nn.LayerNorm(attn_dim)
        
        # ====== Fusion & Classification ======
        self.fusion = nn.Sequential(
            nn.Linear(attn_dim * 2, attn_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(attn_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, snp_data, gm_data):
        """
        Args:
            snp_data: (batch_size, 2098)
            gm_data: (batch_size, 93)
        
        Returns:
            output: (batch_size, output_dim)
        """
        batch_size = snp_data.shape[0]
        
        # ====== SNP 인코더 (feature: batch, 256) ======
        pad_len = self.snp_num_chunks * self.snp_chunk_size - self.snp_input_dim
        if pad_len > 0:
            snp_data = torch.cat([snp_data, torch.zeros(batch_size, pad_len, device=snp_data.device)], dim=1)
        
        snp_x = snp_data.view(batch_size, self.snp_num_chunks, self.snp_chunk_size)
        snp_x = self.snp_mlp(snp_x)  # (batch, 21, 256)
        snp_attn_out, _ = self.snp_attn(snp_x, snp_x, snp_x)  # (batch, 21, 256)
        snp_attn_out = self.snp_norm(snp_attn_out)  # (batch, 21, 256)
        snp_feat = snp_attn_out.mean(dim=1)  # (batch, 256) ← feature!
        
        # ====== GM 인코더 (feature: batch, 256) ======
        gm_x = gm_data.unsqueeze(1)  # (batch, 1, 93)
        gm_x = self.gm_cnn(gm_x)  # (batch, 128, 93)
        gm_x = gm_x.transpose(1, 2)  # (batch, 93, 128)
        gm_x = self.gm_projection(gm_x)  # (batch, 93, 256)
        gm_attn_out, _ = self.gm_attn(gm_x, gm_x, gm_x)  # (batch, 93, 256)
        gm_attn_out = self.gm_norm(gm_attn_out + gm_x)  # (batch, 93, 256)
        gm_feat = gm_attn_out.mean(dim=1)  # (batch, 256) ← feature!
        
        # ====== 양방향 크로스 어텐션 ======
        # GM이 SNP에서 정보 추출
        gm_cross, _ = self.cross_attn_gm_to_snp(
            gm_feat.unsqueeze(1),  # Query: (batch, 1, 256)
            snp_attn_out,  # Key: (batch, 21, 256)
            snp_attn_out   # Value: (batch, 21, 256)
        )
        gm_cross = self.cross_norm1(gm_cross.squeeze(1) + gm_feat)  # (batch, 256)
        
        # SNP가 GM에서 정보 추출
        snp_cross, _ = self.cross_attn_snp_to_gm(
            snp_feat.unsqueeze(1),  # Query: (batch, 1, 256)
            gm_attn_out,  # Key: (batch, 93, 256)
            gm_attn_out   # Value: (batch, 93, 256)
        )
        snp_cross = self.cross_norm2(snp_cross.squeeze(1) + snp_feat)  # (batch, 256)
        
        # ====== Fusion & Classification ======
        fused = torch.cat([snp_cross, gm_cross], dim=1)  # (batch, 512)
        fused = self.fusion(fused)  # (batch, 256)
        output = self.classifier(fused)  # (batch, 4)
        
        return output