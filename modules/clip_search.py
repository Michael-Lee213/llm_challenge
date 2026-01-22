import torch
import open_clip
from PIL import Image
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ClipHit:
    candidate_path: str
    similarity: float

class CLIPSearch:
    def __init__(self, model_id="ViT-B-32/openai", device="cpu", verbose=False):
        self.device = device
        self.verbose = verbose
        m_name, p_name = model_id.split('/')
        self.model, _, self.preproc = open_clip.create_model_and_transforms(
            m_name, pretrained=p_name, device=device
        )
        self.model.eval()
        self.gallery_paths = []
        self.gallery_embs = None

    @torch.no_grad()
    def _embed_image(self, path):
        img = self.preproc(Image.open(path).convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
        return feat.cpu()

    def build_index(self, gallery_dir):
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        # rglob을 사용하여 하위 폴더까지 탐색
        self.gallery_paths = [str(p) for p in Path(gallery_dir).rglob("*") if p.suffix.lower() in exts]
        if not self.gallery_paths:
            raise FileNotFoundError(f"갤러리에 이미지가 없습니다: {gallery_dir}")
        
        if self.verbose: print(f"[CLIP] 인덱싱 중... ({len(self.gallery_paths)}장)")
        embs = [self._embed_image(p) for p in self.gallery_paths]
        # (N, Embedding_Dim) 형태의 텐서 생성
        self.gallery_embs = torch.cat(embs, dim=0)

    def search_with_index(self, query_path, top_k=1):
        q_emb = self._embed_image(query_path) # (1, Dim)
        
        # 코사인 유사도 계산 (N, Dim) @ (Dim, 1) -> (N)
        sims = (self.gallery_embs @ q_emb.T).reshape(-1)
        
        # top_k 추출
        actual_k = min(top_k, len(self.gallery_paths))
        values, indices = sims.topk(actual_k)
        
        # 리스트가 1개일 때도 안전하게 처리
        if actual_k == 1:
            return [ClipHit(self.gallery_paths[indices.item()], float(values.item()))]
            
        return [ClipHit(self.gallery_paths[i], float(v)) for i, v in zip(indices, values)]