from universal_embedding import vit_with_embedding
from universal_embedding import clip_vit_with_embedding
from universal_embedding import udon_vit_with_embedding
from universal_embedding import udon_clip_vit_with_embedding


MODELS = {
  'vit_with_embedding': vit_with_embedding.ViTWithEmbeddingClassificationModel,
  'clip_vit_with_embedding': clip_vit_with_embedding.ViTWithEmbeddingClassificationModel,
  'udon_vit_with_embedding': udon_vit_with_embedding.ViTWithEmbeddingUdonModel,
  'udon_clip_vit_with_embedding': udon_clip_vit_with_embedding.ViTWithEmbeddingUdonModel,
}