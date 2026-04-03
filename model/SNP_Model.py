import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#SNPClassifier 모델 정의. MLP와 어텐션을 활용하여 SNP 데이터를 분류하는 모델.
class SNPClassifier(nn.Module):
    def __init__(self,
                input_dim = 2098, 
                output_dim = 4,
                #SNP 데이터를 처리하기 위한 MLP 레이어에서 사용할 청크 크기 설정. 입력 차원을 청크 크기로 나누어 처리.
                chunk_size = 100,
                #classifier 마지막 레이어 차원 설정. 이후 프로젝션으로 어텐션 레이어 차원으로 맞춰줌.
                classifier_hidden_dim = 256,
                #어텐션 레이어 차원 설정. classifier_hidden_dim과 달라도 상관 없음. projection 레이어에서 맞춰줌.
                attn_dim = 256,
                #어텐션 헤드 수 설정. attn_dim이 num_heads로 나누어 떨어져야 함.
                num_heads = 8,
                ):
        super().__init__()
        #SNP 데이터를 처리하기 위한 MLP 레이어에서 사용할 청크 크기 설정. 입력 차원을 청크 크기로 나누어 처리.
        self.chunk_size = chunk_size
        self.num_chunks = (input_dim + chunk_size - 1) // chunk_size
        self.input_dim = input_dim

        #MLP 레이어 정의. 입력차원을 chunk_size로 설정. 
        #청크 사이즈로 나누어 처리하여 모델의 파라미터 수를 줄임.
        self.SNPclassifier = nn.Sequential(
            nn.Linear(chunk_size, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden_dim, attn_dim)
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

        #SNP 데이터를 MLP 레이어에 입력하기 전에 청크 단위로 나누어 처리. 
        #입력 차원을 chunk_size로 설정하여 모델의 파라미터 수를 줄임.
        batch_size = x.shape[0]

        #num_chunks = 21, chunk_size = 100, input_dim = 2098
        #21 * 100 - 2098 = 2100 - 2098 = 2
        pad_len = self.num_chunks * self.chunk_size - self.input_dim

        #입력 차원을 chunk_size로 나누어 처리하기 위해 패딩 추가. 패딩은 0으로 채움.
        #0을 2개 추가하여 입력 차원을 2100으로 맞춤. 이후 chunk_size로 나누어 처리할 수 있도록 함.
        if pad_len > 0:
            x = torch.cat([x, torch.zeros(batch_size, pad_len, device=x.device)], dim=1)
            
        #
        x = x.view(batch_size, self.num_chunks, self.chunk_size)
        x = self.SNPclassifier(x)
        out, attn_x = self.attn(x, x, x)
        out = self.norm(out)
        out = out.mean(dim=1)
        out = self.classifier(out)
        return out, attn_x


# SNPEncoder 모델 정의. SNP 데이터에서 feature만 추출 (classifier 미적용)
class SNPEncoder(nn.Module):
    def __init__(self,
                input_dim = 2098, 
                chunk_size = 100,
                classifier_hidden_dim = 256,
                attn_dim = 256,
                num_heads = 8,
                ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.num_chunks = (input_dim + chunk_size - 1) // chunk_size
        self.input_dim = input_dim

        self.SNPclassifier = nn.Sequential(
            nn.Linear(chunk_size, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden_dim, attn_dim)
        )

        self.norm = nn.LayerNorm(attn_dim)

        self.attn = nn.MultiheadAttention(
            attn_dim,
            num_heads,
            dropout=0.3,
            batch_first=True
        )

    def forward(self, x):
        batch_size = x.shape[0]
        pad_len = self.num_chunks * self.chunk_size - self.input_dim

        if pad_len > 0:
            x = torch.cat([x, torch.zeros(batch_size, pad_len, device=x.device)], dim=1)
            
        x = x.view(batch_size, self.num_chunks, self.chunk_size)
        x = self.SNPclassifier(x)
        out, attn_x = self.attn(x, x, x)
        out = self.norm(out)
        out = out.mean(dim=1)  # (batch, 256) ← feature만 반환
        return out, attn_x