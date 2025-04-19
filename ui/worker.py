# ui/worker.py
from PyQt6.QtCore import QObject, QRunnable, pyqtSignal

class WorkerSignals(QObject):
    progress = pyqtSignal(str)
    result = pyqtSignal(object)
    finished = pyqtSignal(object)

class FeatureExtractor(QRunnable):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.signals = WorkerSignals()
    
    def run(self):
        hasher = ImageHasher()
        features = {}
        for i, (img_path, img) in enumerate(scan_images(self.root_dir)):
            if self.isInterruptionRequested():
                return
            hash = hasher.dhash(img_path)
            if hash is not None:
                features[img_path] = hash
            self.signals.progress.emit(f"Processing {i+1} images...")
        self.signals.finished.emit(features)

class SimilarityCalculator(QRunnable):
    def __init__(self, feature_db):
        super().__init__()
        self.feature_db = feature_db
        self.signals = WorkerSignals()
    
    def run(self):
        groups = {}
        visited = set()
        for i, (img_path, hash) in enumerate(self.feature_db.items()):
            if img_path in visited:
                continue
            similar = self.find_similar(img_path, hash)
            groups[i] = similar
            visited.update(similar)
            self.signals.progress.emit(f"Grouping {i+1}/{len(self.feature_db)}")
        self.signals.result.emit(groups)
    
    def find_similar(self, target_path, target_hash):
        """查找相似图片"""
        similar = [target_path]
        for img_path, hash in self.feature_db.items():
            if img_path == target_path:
                continue
            distance = self.feature_db.hasher.hamming_distance(target_hash, hash)
            if distance <= self.feature_db.threshold * 256:  # 256 bits
                similar.append(img_path)
        return similar