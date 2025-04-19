# core/hashing.py
import cv2
import numpy as np

class ImageHasher:
    def __init__(self, hash_size=16):
        self.hash_size = hash_size
        
    def process_image(self, img_path):
        """图像预处理流水线"""
        # 读取并转换颜色空间
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None
        
        # 尺寸归一化
        img = cv2.resize(img, (self.hash_size*4, self.hash_size*4))
        
        # 高斯模糊降噪
        img = cv2.GaussianBlur(img, (3,3), 0)
        
        # 直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img)
    
    def dhash(self, img_path):
        """计算差值哈希"""
        processed = self.process_image(img_path)
        if processed is None: return None
        
        # 计算水平梯度差
        diff = processed[:, 1:] > processed[:, :-1]
        return np.packbits(diff.flatten())
    
    def hamming_distance(self, hash1, hash2):
        """计算汉明距离"""
        return np.count_nonzero(np.unpackbits(hash1) != np.unpackbits(hash2))