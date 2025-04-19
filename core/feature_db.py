# core/feature_db.py
import shelve
from pathlib import Path

class FeatureDatabase:
    def __init__(self, db_path='features.db'):
        self.db = shelve.open(db_path)
        self.hasher = ImageHasher()
        self.threshold = 0.85  # 默认阈值
    
    def update(self, new_features):
        """批量更新特征"""
        self.db.update(new_features)
    
    def get(self, img_path):
        return self.db.get(str(img_path))
    
    def items(self):
        return self.db.items()
    
    def __len__(self):
        return len(self.db)
    
    def close(self):
        self.db.close()