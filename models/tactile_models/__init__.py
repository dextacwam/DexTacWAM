# Stage 1 Tactile VAE building blocks and main modules.

from .projector import TactileProjector
from .tactile_modules import (
    FingerCausalConv3d,
    FingerPositionEmbedding,
    ModalityEmbedding,
    PoseTokenizer,
)
from .tactile_vae import (
    TactileEncoder,
    TactileFlowDecoder,
    TactilePoseDecoder,
    TactileVAE,
    TransformerBlock,
    kl_divergence,
    reparameterize,
)

__all__ = [
    # Building blocks (chunk A).
    "FingerCausalConv3d",
    "FingerPositionEmbedding",
    "ModalityEmbedding",
    "PoseTokenizer",
    # Main classes (chunk B).
    "TactileEncoder",
    "TactileFlowDecoder",
    "TactilePoseDecoder",
    "TactileVAE",
    "TransformerBlock",
    # Functional helpers.
    "kl_divergence",
    "reparameterize",
    # Stage 2.
    "TactileProjector",
]
