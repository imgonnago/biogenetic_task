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
                cnn_channels: int = 128,
                #어텐션 레이어 차원 설정. CNN 채널 수와 달라도 상관 없음. projection 레이어에서 맞춰줌.
                attn_dim:int = 256,
                #어텐션 헤드 수 설정. attn_dim이 num_heads로 나누어 떨어져야 함.
                num_heads: int = 8,
                  ):
        super().__init__()

        #CNN 레이어 정의. 1D CNN을 사용하여 GM 데이터를 처리. 채널 수는 cnn_channels로 설정.
        #각 CNN 레이어는 GELU 활성화 함수와 드롭아웃을 포함하여 과적합 방지.
        self.GM_CNN = nn.Sequential(
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

        #CNN 레이어 출력수와 어텐션 레이어 입력 차원을 맞추기 위해 projection 레이어 정의.
        #CNN 레이어의 출력 채널 수인 cnn_channels를 attn_dim으로 변환하여 어텐션 레이어에 입력할 수 있도록 함.
        self.projection = nn.Linear(
            cnn_channels,
            attn_dim
            )
        
        #Normalization 레이어 정의. 
        self.norm = nn.LayerNorm(attn_dim)

        #어텐션 레이어 정의. MultiheadAttention을 사용.
        self.attn = nn.MultiheadAttention(
            attn_dim, 
            num_heads,
            dropout=0.3,
            batch_first=True
            )
        
        #classifier 레이어 정의. 
        self.classifier = nn.Sequential(
            nn.Linear(attn_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )
        
        #forward 정의.
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.GM_CNN(x)
        x = x.transpose(1, 2)
        x = self.projection(x)
        out, atten_w = self.attn(x, x, x)
        out = self.norm(out + x)
        out = out.mean(dim=1)
        x = self.classifier(out) 
        return x, atten_w


# GM_Encoder 모델 정의. GM 데이터에서 feature만 추출 (classifier 미적용)
class GM_Encoder(nn.Module):
    def __init__(self, 
                input_dim: int = 93,
                cnn_channels: int = 128,
                attn_dim: int = 256,
                num_heads: int = 8,
                  ):
        super().__init__()

        self.GM_CNN = nn.Sequential(
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

        self.projection = nn.Linear(
            cnn_channels,
            attn_dim
            )
        
        self.norm = nn.LayerNorm(attn_dim)

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
        out = out.mean(dim=1)  # (batch, 256) ← feature만 반환
        return out, atten_w
    