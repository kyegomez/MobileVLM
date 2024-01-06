import torch
from torch import nn
from zeta.structs import (
    AutoregressiveWrapper,
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)

from mobilevlm.main import LDP


class MobileLLama(torch.nn.Module):

    def __init__(
        self,
        image_size=256,
        patch_size=32,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        alibi_num_heads=4,
        attn_kv_heads=2,
        use_abs_pos_emb=False,
        cross_attend=True,
        alibi_pos_bias=True,
        rotary_xpos=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super(MobileLLama, self).__init__()

        # vit architecture
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, depth=encoder_depth, heads=encoder_heads
            ),
        )
        
        # Mobilevlm
        self.ldp = LDP(
            in_channels=image_size,
            out_channels=image_size,
        )
        
        # 1X1 conv layer to adjust number of channels
        self.conv1 = nn.Conv2d(
            encoder_dim,
            image_size,
            kernel_size=1
        )
        

        # palm model architecture
        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                alibi_pos_bias=alibi_pos_bias,
                alibi_num_heads=alibi_num_heads,
                rotary_xpos=rotary_xpos,
                attn_kv_heads=attn_kv_heads,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )

        # autoregressive wrapper to enable generation of tokens
        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, img: torch.Tensor, text: torch.Tensor):
        """Forward pass of the model."""
        try:
            encoded = self.encoder(img, return_embeddings=True)
            print(f"Encoded shape: {encoded.shape}")
            encoded = self.conv1(encoded)
            print(f"Conv1 shape: {encoded.shape}")
            encoded = self.ldp(encoded)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise

# Usage with random inputs
img = torch.randn(1, 3, 256, 256)
text = torch.randint(0, 20000, (1, 1024))

# Initiliaze the model
model = MobileLLama()
output = model(img, text)
print(output)

