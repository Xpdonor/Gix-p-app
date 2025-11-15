import os, json
import numpy as np

EMB_CACHE = "embeddings.json"
MODEL_CACHE = "ai_model.json"

class AIModule:
    def __init__(self, n_mfcc=20, n_clusters=8):
        self.n_mfcc = n_mfcc
        self.n_clusters = n_clusters
        self.path_to_emb = {}
        self.cluster_labels = {}
        self.cluster_centers = None

    def extract_embedding(self, path):
        """استخراج ویژگی‌های صدا بدون librosa"""
        try:
            import soundfile as sf
            y, sr = sf.read(path)
            
            if y.ndim > 1:
                y = y.mean(axis=1)
            
            # استخراج MFCC ساده
            mfcc = self.simple_mfcc(y, sr, n_mfcc=self.n_mfcc)
            emb = np.mean(mfcc, axis=1)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            return emb.astype(np.float32)
        except Exception as e:
            print("emb fail", path, e)
            return None

    def simple_mfcc(self, y, sr, n_mfcc=20):
        """پیاده‌سازی ساده MFCC بدون librosa"""
        # تبدیل فوریه کوتاه-زمان (STFT)
        frame_length = 2048
        hop_length = 512
        
        stft = np.array([np.fft.rfft(y[i:i+frame_length] * np.hanning(frame_length))
                        for i in range(0, len(y)-frame_length, hop_length)])
        
        # طیف توان
        power_spectrum = np.abs(stft) ** 2
        
        # فیلترهای مل (ساده‌شده)
        n_mels = 40
        mel_filters = self.create_mel_filterbank(sr, frame_length, n_mels)
        
        # اعمال فیلترهای مل
        mel_spectrum = np.dot(power_spectrum, mel_filters.T)
        mel_spectrum = np.log(mel_spectrum + 1e-9)
        
        # DCT (MFCC)
        mfcc = np.array([np.fft.dct(frame)[:n_mfcc] for frame in mel_spectrum])
        
        return mfcc.T

    def create_mel_filterbank(self, sr, n_fft, n_mels):
        """ایجاد فیلترهای مل ساده"""
        low_freq = 0
        high_freq = sr / 2
        
        # تبدیل فرکانس به مقیاس مل
        low_mel = 2595 * np.log10(1 + low_freq / 700)
        high_mel = 2595 * np.log10(1 + high_freq / 700)
        
        mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
        freq_points = 700 * (10 ** (mel_points / 2595) - 1)
        
        # ایجاد فیلترها
        bins = np.floor((n_fft + 1) * freq_points / sr).astype(int)
        filters = np.zeros((n_mels, n_fft // 2 + 1))
        
        for i in range(1, n_mels + 1):
            left = bins[i-1]
            center = bins[i]
            right = bins[i+1]
            
            if left < center:
                filters[i-1, left:center] = np.linspace(0, 1, center - left)
            if center < right:
                filters[i-1, center:right] = np.linspace(1, 0, right - center)
        
        return filters

    def simple_kmeans(self, data, n_clusters, max_iters=100):
        """K-means ساده"""
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

    def build_index(self, paths):
        if os.path.exists(EMB_CACHE):
            try:
                data = json.load(open(EMB_CACHE,'r',encoding='utf-8'))
                for it in data:
                    self.path_to_emb[it['path']] = np.array(it['emb'], dtype=np.float32)
            except Exception:
                pass
        
        missing = [p for p in paths if p not in self.path_to_emb]
        for p in missing:
            e = self.extract_embedding(p)
            if e is not None:
                self.path_to_emb[p] = e.tolist()
        
        dump = [{'path':p,'emb':self.path_to_emb[p] if isinstance(self.path_to_emb[p],list) else self.path_to_emb[p].tolist()} for p in self.path_to_emb]
        json.dump(dump, open(EMB_CACHE,'w',encoding='utf-8'), ensure_ascii=False)
        
        embs = np.array([np.array(x['emb'], dtype=np.float32) for x in dump])
        if len(embs) < 2:
            return
        
        n_clusters = min(self.n_clusters, len(embs))
        labels, centers = self.simple_kmeans(embs, n_clusters)
        self.cluster_centers = centers
        
        for i, it in enumerate(dump):
            self.cluster_labels[it['path']] = int(labels[i])
        
        json.dump({'clusters': int(self.n_clusters)}, open(MODEL_CACHE,'w'))

    def get_cluster(self, path):
        return self.cluster_labels.get(path, None)

    def recommend_eq(self, path, output='speaker'):
        emb = self.path_to_emb.get(path)
        if emb is None:
            emb = self.extract_embedding(path)
            if emb is None:
                return {'low': 0.0, 'mid': 0.0, 'high': 0.0}
        
        # محاسبه centroid ساده
        try:
            import soundfile as sf
            y, sr = sf.read(path)
            if y.ndim > 1:
                y = y.mean(axis=1)
            
            # طیف توان ساده
            spectrum = np.abs(np.fft.rfft(y[:8192]))
            freqs = np.fft.rfftfreq(8192, 1/sr)
            sc = np.sum(spectrum * freqs) / np.sum(spectrum)
        except Exception:
            sc = 2000.0
        
        if sc < 2000:
            base = {'low': -0.5, 'mid': 0.0, 'high': 1.0}
        else:
            base = {'low': 1.0, 'mid': 0.0, 'high': -0.5}
        
        cl = self.get_cluster(path)
        if cl is not None:
            tweak = ((cl % 3) - 1) * 0.6
            base['low'] += tweak
            base['high'] -= tweak * 0.5
        
        if output == 'headphones':
            base['low'] *= 0.6
            base['high'] *= 1.1
        else:
            base['low'] *= 1.2
            base['high'] *= 0.9
        
        return base
