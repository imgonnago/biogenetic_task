import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F 
from typing import Optional

# GM_CNN 모델 정의. CNN과 어텐션을 활용하여 GM 데이터를 분류하는 모델.
class GM_CNN(nn.Module):
    def __init__(self, 
                input_dim: int = 93,
                output_dim: int = 4,
                #CNN 마지막 채널수 설정. 이후 프로젝션으로 어텐션 레이어 차원으로 맞춰줌.
                cnn_channels: int = 64,
                #어텐션 레이어 차원 설정. CNN 채널 수와 달라도 상관 없음. projection 레이어에서 맞춰줌.
                attn_dim:int = 64,
                #어텐션 헤드 수 설정. attn_dim이 num_heads로 나누어 떨어져야 함.
                num_heads: int = 8,
                  ):
        super().__init__()

        # [수정] Conv 3층으로 유지하되 채널 확대: 1→32→64→cnn_channels
        # kernel_size 5→3→3: 첫 레이어에서 더 넓은 패턴 포착
        # 기존: 1→16→32→64 (채널 작음), dropout 1개
        # 수정: 1→32→64→cnn_channels, dropout 제거 (93개 피처에 dropout 불필요)
        self.GM_CNN = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.Dropout(0.2),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
        )

        #CNN 레이어 출력수와 어텐션 레이어 입력 차원을 맞추기 위해 projection 레이어 정의.
        self.projection = nn.Linear(
            cnn_channels,
            attn_dim
            )
        
        #Normalization 레이어 정의. 
        self.norm = nn.LayerNorm(attn_dim)

        # [수정] attention dropout 0.2 → 0.05
        self.attn = nn.MultiheadAttention(
            attn_dim, 
            num_heads,
            dropout=0.1,
            batch_first=True
            )
        
        # [수정] classifier 용량 확대: attn_dim→64→32→output
        # 기존: attn_dim→32→output (중간 레이어 1개로 표현력 부족)
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.GM_CNN(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        out, atten_w = self.attn(x, x, x)
        out = self.norm(out + x)
        out = out.max(dim=1).values
        x = self.classifier(out) 
        return x, atten_w


# GM_Encoder 모델 정의. GM 데이터에서 feature만 추출 (classifier 미적용)
class GM_Encoder(nn.Module):
    def __init__(self, 
                input_dim: int = 93,
                cnn_channels: int = 64,
                attn_dim: int = 64,
                num_heads: int = 8,
                  ):
        super().__init__()

        # [수정] Conv 4층 → 3층으로 축소, 채널 확대, dropout 제거
        # 기존: 1→32→64→64→64, dropout×2 (93개 피처에 4층은 과도)
        # 수정: 1→32→64→cnn_channels, dropout 없음
        # kernel_size 5→3→3: 첫 레이어에서 더 넓은 수용 영역 확보
        self.GM_CNN = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, cnn_channels, kernel_size=3, padding=1),
        )

        self.projection = nn.Linear(
            cnn_channels,
            attn_dim
            )
        
        self.norm = nn.LayerNorm(attn_dim)

        # [수정] attention dropout 0.1 → 0.05
        self.attn = nn.MultiheadAttention(
            attn_dim, 
            num_heads,
            dropout=0.3,
            batch_first=True
            )
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.GM_CNN(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        out, atten_w = self.attn(x, x, x)
        out = self.norm(out + x)
        out = out.mean(dim=1)
        return out, atten_w
