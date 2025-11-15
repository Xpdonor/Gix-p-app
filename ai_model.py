import os, json
import numpy as np
import librosa

EMB_CACHE = "embeddings.json"
MODEL_CACHE = "ai_model.json"


class AIModule:
    def __init__(self, n_mfcc=20, n_clusters=8):
        self.n_mfcc = n_mfcc
        self.n_clusters = n_clusters
        self.path_to_emb = {}
        self.cluster_labels = {}
        self.cluster_centers = None
        self.pca_components = None
        self.pca_mean = None

    def extract_embedding(self, path):
        try:
            y, sr = librosa.load(path, sr=22050, mono=True, duration=60)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            emb = np.mean(mfcc, axis=1)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            return emb.astype(np.float32)
        except Exception as e:
            print("emb fail", path, e)
            return None

    def simple_pca(self, data, n_components):

        mean = np.mean(data, axis=0)
        centered = data - mean


        cov = np.cov(centered, rowvar=False)


        eigenvalues, eigenvectors = np.linalg.eigh(cov)


        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = eigenvalues[idx]


        components = eigenvectors[:, :n_components]

        return components, mean

    def simple_kmeans(self, data, n_clusters, max_iters=100):

        n_samples = data.shape[0]


        centers = data[np.random.choice(n_samples, n_clusters, replace=False)]

        for _ in range(max_iters):

            distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)


            labels = np.argmin(distances, axis=1)


            new_centers = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
                                    for i in range(n_clusters)])


            if np.allclose(centers, new_centers):
                break

            centers = new_centers

        return labels, centers

    def pca_transform(self, data, components, mean):

        centered = data - mean
        return np.dot(centered, components)

    def build_index(self, paths):
        # try cache
        if os.path.exists(EMB_CACHE):
            try:
                data = json.load(open(EMB_CACHE, 'r', encoding='utf-8'))
                for it in data:
                    self.path_to_emb[it['path']] = np.array(it['emb'], dtype=np.float32)
            except Exception:
                pass

        # compute missing
        missing = [p for p in paths if p not in self.path_to_emb]
        for p in missing:
            e = self.extract_embedding(p)
            if e is not None:
                self.path_to_emb[p] = e.tolist()

        # write cache
        dump = [{'path': p,
                 'emb': self.path_to_emb[p] if isinstance(self.path_to_emb[p], list) else self.path_to_emb[p].tolist()}
                for p in self.path_to_emb]
        json.dump(dump, open(EMB_CACHE, 'w', encoding='utf-8'), ensure_ascii=False)

        # clustering
        embs = np.array([np.array(x['emb'], dtype=np.float32) for x in dump])
        if len(embs) < 2:
            return

        # PCA reduction
        n_components = min(8, embs.shape[1])
        self.pca_components, self.pca_mean = self.simple_pca(embs, n_components)
        reduced = self.pca_transform(embs, self.pca_components, self.pca_mean)

        # K-means clustering
        n_clusters = min(self.n_clusters, len(reduced))
        labels, centers = self.simple_kmeans(reduced, n_clusters)

        self.cluster_centers = centers

        for i, it in enumerate(dump):
            self.cluster_labels[it['path']] = int(labels[i])

        # persist model metadata
        json.dump({'clusters': int(self.n_clusters)}, open(MODEL_CACHE, 'w'))

    def get_cluster(self, path):
        return self.cluster_labels.get(path, None)

    def recommend_eq(self, path, output='speaker'):
        
        emb = self.path_to_emb.get(path)
        if emb is None:
            emb = self.extract_embedding(path)
            if emb is None:
                return {'low': 0.0, 'mid': 0.0, 'high': 0.0}

        # spectral centroid fallback
        try:
            y, sr = librosa.load(path, sr=22050, mono=True, duration=30)
            sc = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        except Exception:
            sc = 2000.0

        # heuristic mapping
        if sc < 2000:
            base = {'low': -0.5, 'mid': 0.0, 'high': 1.0}
        else:
            base = {'low': 1.0, 'mid': 0.0, 'high': -0.5}

        # cluster-based tweak
        cl = self.get_cluster(path)
        if cl is not None:
            tweak = ((cl % 3) - 1) * 0.6
            base['low'] += tweak
            base['high'] -= tweak * 0.5

        # output device adjust
        if output == 'headphones':
            base['low'] *= 0.6
            base['high'] *= 1.1
        else:  # speaker
            base['low'] *= 1.2
            base['high'] *= 0.9

        return base