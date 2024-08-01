import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout: int = 0.3, qk_norm: bool = True):
        super(MultiModalCrossAttention, self).__init__()

        self.num_heads = num_heads
        self.dim = dim
        self.dk = dim // num_heads
        self.qk_norm = qk_norm

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

        # Query, Key, Value projection layers for video -> audio

        self.Wq = nn.Linear(dim, dim/2)
        self.Wk = nn.Linear(dim, dim/2)
        self.Wv = nn.Linear(dim, dim/2)

        # Query, Key, Value projection layers for audio -> video
        self.Wq_reverse = nn.Linear(dim, dim)
        self.Wk_reverse = nn.Linear(dim, dim)
        self.Wv_reverse = nn.Linear(dim, dim)

        # Output linear layer after attention computation
        self.linear_out = nn.Linear(2 * dim, dim)

    def forward(self, v_feature, a_feature):
        # video -> audio
        Qcross = self.Wq(v_feature)
        Kcross = self.Wk(a_feature)
        Vcross = self.Wv(a_feature)

        if self.qk_norm:
            # Normalize Qcross and Kcross
            Qcross = self.norm(Qcross)
            Kcross = self.norm(Kcross)
        else:
            pass

        with torch.backends.cuda.sdp_kernel(enable_math=True):
            attn_weights = F.scaled_dot_product_attention(Qcross, Kcross, Vcross)

            attn_weights = self.dropout(attn_weights)

        print(
            f"attn_weights shape: {attn_weights.shape}, and vcross shape: {Vcross.shape}"
        )

        Videocross = torch.matmul(attn_weights, Vcross)

        # audio -> video
        Qcross_reverse = self.Wq_reverse(a_feature)
        Kcross_reverse = self.Wk_reverse(v_feature)
        Vcross_reverse = self.Wv_reverse(v_feature)

        with torch.backends.cuda.sdp_kernel(enable_math=True):
            attn_weights_reverse = F.scaled_dot_product_attention(
                Qcross_reverse, Kcross_reverse, Vcross_reverse
            )

            attn_weights_reverse = self.dropout(attn_weights_reverse)

        Audiocross = torch.matmul(attn_weights_reverse, Vcross_reverse)

        output = torch.cat((Videocross, Audiocross), dim=-1)

        output = self.linear_out(output)

        return output

if __name__ == '__main__':
    dim = 512
    num_heads = 8
    cross_attn = MultiModalCrossAttention(dim, num_heads)
    video_sample = torch.randn(32, dim, dim)
    audio_sample = torch.randn(32, dim, dim)
    
    output = cross_attn(video_sample, audio_sample)
    print(output)
    print(output.shape)
