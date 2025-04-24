import random
import cv2
import numpy as np
import os
from collections import defaultdict

class SimilarityAlgorithms:
    def __init__(self, group_data, add_group_callback, init_group_tree_callback):
        """
        初始化 SimilarityAlgorithms 类。

        :param group_data: 当前的分组数据（字典）
        :param add_group_callback: 添加分组的回调函数
        :param init_group_tree_callback: 刷新分组树的回调函数
        """
        self.group_data = group_data
        self.add_group_callback = add_group_callback
        self.init_group_tree_callback = init_group_tree_callback

    @staticmethod
    def load_image_with_chinese_path(img_path):
        with open(img_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"图像加载失败，路径可能无效: {img_path}")
        return img

    def apply_single_algorithm_grouping(self, images, threshold=0.85, algorithm="phash"):

        # 路径规范化与去重
        images = [os.path.abspath(img) for img in images]
        images = list({os.path.normcase(img) for img in images})
        
        # 预计算所有特征
        feature_cache = {}
        for img in images:
            if algorithm == "phash":
                feature_cache[img] = self.pHash(img)
            elif algorithm == "histogram":
                feature_cache[img] = self._calc_histogram(img)
        print(f"特征缓存内容: {feature_cache}")  # 正确显示缓存内容
        # 两两比对
        groups = {}
        grouped = set()
        for i, img1 in enumerate(images):
            if img1 in grouped: continue
            
            current_group = [img1]
            for img2 in images[i+1:]:
                if algorithm == "phash":
                    sim = self.pHash_similarity(feature_cache[img1], feature_cache[img2])
                elif algorithm == "histogram":
                    sim = 1 - cv2.compareHist(feature_cache[img1], feature_cache[img2], 
                                            cv2.HISTCMP_BHATTACHARYYA)
                elif algorithm == "sift":
                    sim = self.sift_similarity(img1, img2)
                else:  # combined
                    sim = self.combined_similarity(img1, img2)
                
                if sim >= threshold:
                    current_group.append(img2)
                    grouped.add(img2)
            
            # 创建动态分组名
            base_name = os.path.basename(img1)
            group_name = f"{algorithm}_{base_name[:10]}_group"
            groups[group_name] = current_group
        
        return groups

    @staticmethod
    def _calc_histogram(img_path):
        """计算图像的直方图"""
        try:
            # 使用支持中文路径的加载方法
            img = SimilarityAlgorithms.load_image_with_chinese_path(img_path)
        except Exception as e:
            raise ValueError(f"图像加载失败，路径可能无效: {img_path}") from e

        # 计算直方图逻辑（保持不变）
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def random_grouping(self, images, num_groups=5):
            """
            将图片随机分成指定数量的组。

            :param images: 图片路径列表
            :param num_groups: 分组数量，默认为 5
            :return: 一个字典，键为组名，值为图片路径列表
            """
            import random
            groups = {f"group_{i+1}": [] for i in range(num_groups)}
            for image in images:
                group_name = random.choice(list(groups.keys()))
                groups[group_name].append(image)
            return groups
        
    
    def pHash(self, image_path):
        """改进的感知哈希算法，支持中文路径[1](@ref)"""
        try:
            img = self._load_image(image_path)
            if img is None:
                return None
                
            # 统一处理为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
            
            # DCT变换和哈希生成
            dct = cv2.dct(np.float32(resized))
            roi = dct[0:8, 0:8]
            avg = np.mean(roi)
            hash_value = ''.join(['1' if x > avg else '0' for x in roi.flatten()])
            print(f"Hash for {image_path}: {hash_value}")
            return hash_value
        except Exception as e:
            print(f"pHash计算失败: {str(e)}")
            return None

    @staticmethod
    def pHash_similarity(hash1, hash2):
        """汉明距离计算相似度[3](@ref)"""
        if len(hash1) != len(hash2):
            return 0.0
        hamming = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        return 1 - hamming / len(hash1)
    
    @staticmethod
    def _load_image(path):
        """统一图像加载方法，支持中文路径[1,4](@ref)"""
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"文件不存在: {path}")
            img_data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"图像解码失败: {path}")
            return img
        except Exception as e:
            print(f"图像加载失败: {str(e)}")
            return None

    def histogram_similarity(self, img1_path, img2_path):
        """优化后的颜色直方图相似度计算[1,3,4](@ref)"""
        try:
            img1 = self._load_image(img1_path)
            img2 = self._load_image(img2_path)
            if img1 is None or img2 is None:
                return 1.0  # 返回最大差异值

            # 转换到HSV颜色空间[4](@ref)
            img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

            # 计算2D直方图（H和S通道）[3](@ref)
            hist1 = cv2.calcHist([img1_hsv], [0, 1], None, 
                               [180, 256], [0, 180, 0, 256])
            hist2 = cv2.calcHist([img2_hsv], [0, 1], None,
                               [180, 256], [0, 180, 0, 256])

            # 归一化处理[5](@ref)
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            # 使用巴氏距离比较直方图[4](@ref)
            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
        except Exception as e:
            print(f"直方图计算异常: {str(e)}")
            return 1.0  # 默认返回最大差异值
    
    @staticmethod
    def sift_similarity(img1_path, img2_path):
        """SIFT特征匹配算法"""
        try:
            # 使用支持中文路径的加载方法
            img1 = SimilarityAlgorithms.load_image_with_chinese_path(img1_path)
            img2 = SimilarityAlgorithms.load_image_with_chinese_path(img2_path)

            # 检查图像是否加载成功
            if img1 is None or img2 is None:
                return 0.0  # 返回默认相似度

            # 特征检测
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            if des1 is None or des2 is None:
                return 0.0  # 如果无法提取特征，返回默认相似度

            # 特征匹配
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # Lowe's比率测试
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            # 相似度计算
            similarity = len(good) / max(len(kp1), len(kp2))
            return float(similarity)  # 确保返回值为浮点数
        except Exception as e:
            print(f"SIFT相似度计算失败: {str(e)}")
            return 0.0  # 异常时返回默认相似度
    

    def apply_combined_grouping(self, images, 
                                coarse_threshold=0.7,
                                fine_threshold=0.6):
        """两阶段相似度分组算法"""
        coarse_threshold = float(coarse_threshold)
        fine_threshold = float(fine_threshold)

        # 特征预计算
        feature_cache = {
            img: {
                'phash': self.pHash(img),
                'hist': self._calc_histogram(img)
            } for img in images if os.path.exists(img)
        }

        # 初始化分组结果和已分组集合
        final_groups = {}
        grouped = set()

        # 遍历所有图片
        for img1 in images:
            if img1 in grouped or img1 not in feature_cache:
                continue

            # 当前组初始化
            current_group = set([img1])
            queue = [img1]  # 使用队列进行递归检查

            while queue:
                base_img = queue.pop(0)
                for img2 in images:
                    if img2 in grouped or img2 not in feature_cache or img2 == base_img:
                        continue

                    # 计算相似度
                    phash_sim = self.pHash_similarity(
                        feature_cache[base_img]['phash'],
                        feature_cache[img2]['phash']
                    )
                    hist_sim = 1 - cv2.compareHist(
                        feature_cache[base_img]['hist'],
                        feature_cache[img2]['hist'],
                        cv2.HISTCMP_BHATTACHARYYA
                    )
                    combined_sim = 0.6 * phash_sim + 0.4 * hist_sim

                    if combined_sim >= coarse_threshold:
                        current_group.add(img2)
                        queue.append(img2)
                        grouped.add(img2)

            # 第二阶段：细粒度验证
            valid_group = []
            for img in current_group:
                if all(self.sift_similarity(img, other) >= fine_threshold for other in current_group if img != other):
                    valid_group.append(img)

            # 如果细粒度验证失败，将图片单独分组
            if not valid_group:
                for img in current_group:
                    group_name = f"single_{os.path.basename(img)[:6]}_group"
                    final_groups[group_name] = [img]
                    grouped.add(img)
            else:
                # 添加到最终分组
                group_name = f"group_{len(final_groups) + 1}"
                final_groups[group_name] = valid_group
                grouped.update(valid_group)

        # 处理未分组的图片
        ungrouped_images = [img for img in images if img not in grouped]
        for img in ungrouped_images:
            group_name = f"single_{os.path.basename(img)[:6]}_group"
            final_groups[group_name] = [img]

        return final_groups

    def combined_similarity(self, img1_path, img2_path, 
                            coarse_threshold=0.7, 
                            fine_threshold=0.6,
                            weights=(0.6, 0.4)):
        """两阶段混合相似度检测算法"""
        try:
            # 第一阶段：粗粒度筛选（pHash + 直方图）
            hash1 = self.pHash(img1_path)
            hash2 = self.pHash(img2_path)
            phash_sim = self.pHash_similarity(hash1, hash2) if hash1 and hash2 else 0
            
            hist_sim = 1 - self.histogram_similarity(img1_path, img2_path)  # 转换为相似度
            
            # 加权计算粗粒度相似度 [1,3](@ref)
            coarse_sim = weights[0]*phash_sim + weights[1]*hist_sim
            
            if coarse_sim < coarse_threshold:
                return 0.0  # 未通过粗筛直接返回
            
            # 第二阶段：细粒度验证（SIFT）
            sift_sim = self.sift_similarity(img1_path, img2_path)
            return sift_sim if sift_sim >= fine_threshold else 0.0

        except Exception as e:
            print(f"混合算法异常: {str(e)}")
            return 0.0