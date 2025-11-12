
import json, os
import numpy as np
from collections import OrderedDict
from analyzer import spectral_centroid
from ai_model import AIModule

class MusicIndex:
    def __init__(self, paths):
        self.paths = paths
        self.embs = OrderedDict()
        self.meta = {}
        self._build_index()

    def _build_index(self):
        from mutagen import File as MutagenFile
        for p in self.paths:
            # try load embedding from embeddings.json if present
            # simple: call AIModule.extract_embedding? but we used ai_model caching
            try:
                # metadata
                tag = MutagenFile(p)
                title = os.path.basename(p)
                artist = 'Unknown'
                album = 'Unknown'
                if tag and getattr(tag,'tags',None):
                    try:
                        if 'TIT2' in tag.tags:
                            title = tag.tags['TIT2'].text[0]
                    except:
                        pass
                self.meta[p] = {'title': title, 'artist': artist, 'album': album, 'cover': None}
            except Exception:
                self.meta[p] = {'title': os.path.basename(p), 'artist': 'Unknown', 'album': 'Unknown', 'cover': None}
        # try load embeddings cache
        if os.path.exists('embeddings.json'):
            try:
                data = json.load(open('embeddings.json','r',encoding='utf-8'))
                for it in data:
                    self.embs[it['path']] = np.array(it['emb'], dtype=np.float32)
            except Exception:
                pass

    def most_similar(self, path, top_k=6):
        if path not in self.embs:
            return []
        q = self.embs[path]
        sims = []
        for p,e in self.embs.items():
            if p == path: continue
            sims.append((float(np.dot(q, e)), p))
        sims.sort(reverse=True)
        return [p for _,p in sims[:top_k]]
