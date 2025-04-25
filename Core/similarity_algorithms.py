import cv2
import numpy as np
import os
import sys
import gc
import random
from collections import defaultdict
from PyQt6.QtCore import QObject, pyqtSignal, QRunnable, QMutex, QMutexLocker
from concurrent.futures import ThreadPoolExecutor

class SimilaritySignals(QObject):
    progress_updated = pyqtSignal(int)
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

class SimilarityWorker(QRunnable):
    def __init__(self, algorithm, images, coarse_threshold, fine_threshold, similarity_algo):
        super().__init__()
        self.algorithm = algorithm
        self.images = images
        self.coarse_threshold = coarse_threshold
        self.fine_threshold = fine_threshold
        self.similarity_algo = similarity_algo
        self.signals = SimilaritySignals()
        self._is_running = True

    def run(self):
        #try:
        if self.algorithm == "combined":
            results = self.similarity_algo.apply_combined_grouping(
                self.images,
                self.coarse_threshold,
                self.fine_threshold,
                progress_callback=self.signals.progress_updated.emit
            )
        else:
            results = self.similarity_algo.apply_single_algorithm_grouping(
                self.images,
                threshold=self.similarity_algo.main_app.ui_manager.similarity_threshold,
                algorithm=self.algorithm,
                progress_callback=self.signals.progress_updated.emit
            )
            
        if self._is_running:
            self.signals.result_ready.emit(results)
                
        #except Exception as e:
            #self.signals.error_occurred.emit(f"计算失败: {str(e)}")

    def cancel(self):
        self._is_running = False

class SimilarityAlgorithms:
    def __init__(self, group_data, add_group_callback, init_group_tree_callback):
        self.group_data = group_data
        self.add_group_callback = add_group_callback
        self.init_group_tree_callback = init_group_tree_callback
        self.main_app = None  
        self.group_mutex = QMutex()  # 新增互斥锁
        self.cache_mutex = QMutex()

    def apply_single_algorithm_grouping(self, images, threshold=0.85, algorithm="phash", progress_callback=None):
        total = len(images)
        feature_cache = {}
        valid_images = []
        
        # 阶段1：带进度的特征预计算
        for i, img_path in enumerate(images):
            try:
                if progress_callback:
                    progress = int((i / total) * 50)
                    progress_callback(progress)

                # 特征计算逻辑
                if algorithm == "phash":
                    feature = self.pHash(img_path)
                elif algorithm == "histogram":
                    feature = self._calc_histogram(img_path)
                elif algorithm == "sift":
                    feature = self.sift_descriptor(img_path)
                else:
                    raise ValueError(f"不支持的算法类型: {algorithm}")

                if feature is not None:
                    feature_cache[img_path] = feature
                    valid_images.append(img_path)
            except Exception as e:
                print(f"特征计算失败 {img_path}: {str(e)}")

        # 阶段2：带进度的分组计算
        groups = defaultdict(list)
        processed = set()
        for idx, img1 in enumerate(valid_images):
            if img1 in processed:
                continue
            
            current_group = [img1]
            processed.add(img1)
            base_feature = feature_cache[img1]
            
            # 相似度比较
            for img2 in valid_images[idx+1:]:
                if img2 in processed:
                    continue
                
                # 相似度计算
                if algorithm == "phash":
                    sim = self.pHash_similarity(base_feature, feature_cache[img2])
                elif algorithm == "histogram":
                    sim = 1 - cv2.compareHist(base_feature, feature_cache[img2], cv2.HISTCMP_BHATTACHARYYA)
                elif algorithm == "sift":
                    sim = self.sift_similarity(img1, img2)
                
                if sim >= threshold:
                    current_group.append(img2)
                    processed.add(img2)

            # 生成动态组名
            group_name = f"{algorithm}_group_{len(groups)+1}"
            groups[group_name] = current_group
            
            # 进度更新
            if progress_callback:
                progress = 50 + int(((idx + 1) / len(valid_images)) * 50)
                progress_callback(progress)

        return groups
    def apply_combined_grouping(self, images, coarse_threshold, fine_threshold, progress_callback=None):
        with QMutexLocker(self.group_mutex):
            total = len(images)
            feature_cache = {}
            valid_images = []
            global_candidate_pool = set()

            max_workers = max(2, os.cpu_count() - 1)
            feature_load_executor = ThreadPoolExecutor(max_workers=max_workers)

            # 阶段1：特征预计算
            with feature_load_executor as executor:
                sift_preload_futures = []
                for i, img_path in enumerate(images):
                    if progress_callback and i % 10 == 0:
                        progress = int((i / total) * 30)
                        progress_callback(progress)
                    
                    future = executor.submit(self._precompute_features_with_sift, img_path, feature_cache)
                    sift_preload_futures.append(future)
                
                for future in sift_preload_futures:
                    try:
                        img_path, features = future.result()
                        if not isinstance(features, dict):
                            raise ValueError(f"Invalid features for {img_path}: {type(features)}")
                        valid_images.append(img_path)
                        global_candidate_pool.add(img_path)
                    except Exception as e:
                        print(f"特征预计算失败: {str(e)}")

            # 内存管理
            if sys.getsizeof(feature_cache) > 100 * 1024 * 1024:
                for img in list(feature_cache.keys())[::2]:
                    del feature_cache[img]
                gc.collect()

            final_groups = defaultdict(list)
            processed = set()
            recovery_threshold = fine_threshold * 0.8

            # 阶段2：主处理流程
            for idx, img1 in enumerate(valid_images):
                if img1 in processed:
                    continue

                if progress_callback and idx % 5 == 0:
                    current_progress = 30 + int((idx / len(valid_images)) * 60)
                    progress_callback(current_progress)

                temp_processed = {img1}
                coarse_group = [img1]
                base_features = feature_cache[img1]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    batch_size = 50
                    img_batch = []

                    for img2 in valid_images[idx + 1:]:
                        if img2 not in processed and img2 not in temp_processed:
                            img_batch.append(img2)
                            if len(img_batch) == batch_size:
                                    futures.append(executor.submit(
                                    self._batch_cluster_similarity,
                                    base_features,                # cluster参数（基准特征）
                                    img_batch,                    # candidates参数（图像路径列表）
                                    feature_cache,               # 正确传入特征缓存字典
                                    coarse_threshold             # 阈值参数
                                ))
                            img_batch = []

                    if img_batch:
                        futures.append(executor.submit(
                            self._batch_cluster_similarity,
                            base_features,                # cluster参数（基准特征）
                            img_batch,                    # candidates参数（图像路径列表）
                            feature_cache,               # 正确传入特征缓存字典
                            coarse_threshold             # 阈值参数
                        ))

                    for future in futures:
                        matched_imgs = future.result()
                        coarse_group.extend(matched_imgs)
                        temp_processed.update(matched_imgs)
                        global_candidate_pool.difference_update(matched_imgs)

                validation_matrix = defaultdict(set)
                sift_features = {}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {img: executor.submit(self._get_sift_features, img, feature_cache)
                            for img in coarse_group}
                    for img, future in futures.items():
                        sift_features[img] = future.result()

                ransac_threshold = 8.0
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    match_futures = []
                    for i, img_a in enumerate(coarse_group):
                        for img_b in coarse_group[i + 1:]:
                            match_futures.append(executor.submit(
                                self.sift_similarity,
                                img_a, img_b,
                                sift_features[img_a],
                                sift_features[img_b],
                                ransac_threshold
                            ))

                    for i, future in enumerate(match_futures):
                        img_a, img_b = coarse_group[i // (len(coarse_group) - 1)], coarse_group[i % (len(coarse_group) - 1) + 1]
                        sim = future.result()
                        if sim >= fine_threshold:
                            validation_matrix[img_a].add(img_b)
                            validation_matrix[img_b].add(img_a)

                max_cluster = self._find_max_cluster(coarse_group, validation_matrix)

                if max_cluster:
                    recovery_candidates = []
                    candidate_group = list(global_candidate_pool - set(max_cluster))

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        batch_size = 20
                        candidate_batches = [candidate_group[i:i + batch_size]
                                            for i in range(0, len(candidate_group), batch_size)]

                        futures = []
                        for batch in candidate_batches:
                            futures.append(executor.submit(
                                self._batch_cluster_similarity,
                                max_cluster,
                                batch,
                                feature_cache,
                                recovery_threshold
                            ))

                        for future in futures:
                            recovery_candidates.extend(future.result())

                    if recovery_candidates:
                        max_cluster.extend(recovery_candidates)
                        processed.update(recovery_candidates)
                        global_candidate_pool -= set(recovery_candidates)

                    group_name = f"combined_group_{len(final_groups) + 1}"
                    final_groups[group_name] = max_cluster
                    processed.update(max_cluster)

                if progress_callback and idx % 10 == 0:
                    progress = 90 + int(((idx + 1) / len(valid_images)) * 10)
                    progress_callback(progress)

            return {str(k): [str(p) for p in v] for k, v in final_groups.items()}


    def _precompute_features_with_sift(self, img_path, feature_cache):
        with QMutexLocker(self.cache_mutex):  # 加锁操作
            try:
                # 初始化字典结构
                if img_path not in feature_cache:
                    feature_cache[img_path] = {'phash': None, 'hist': None, 'sift': None}
                
                # 并行加载特征
                with ThreadPoolExecutor(max_workers=2) as executor:
                    phash_future = executor.submit(self.pHash, img_path)
                    hist_future = executor.submit(self._calc_histogram, img_path)
                    sift_future = executor.submit(self.sift_descriptor, img_path)

            
                feature_cache[img_path].update({
                    'phash': phash_future.result(),
                    'hist': hist_future.result(),
                    'sift': sift_future.result()
                })
                return img_path, feature_cache[img_path]
            except Exception as e:
                print(f"特征预计算失败 {img_path}: {str(e)}")
                # 确保返回结构一致性
                return img_path, {'phash': None, 'hist': None, 'sift': None}

    def _batch_cluster_similarity(self, cluster, candidates, feature_cache, threshold):
        """批量候选相似度计算"""

        valid_candidates = []
        
        # 类型安全检查
        if not isinstance(feature_cache, dict):
            raise ValueError("feature_cache必须是字典类型")
            
        for candidate in candidates:
            # 获取候选图像特征
            candidate_features = feature_cache.get(candidate)
            if not candidate_features or 'sift' not in candidate_features:
                continue
                
            candidate_sift = candidate_features['sift']
            total_sim = 0.0
            valid_comparisons = 0
            
            # 随机采样基准图像(防止cluster过大)
            sample_size = min(5, len(cluster))
            sample_members = random.sample(cluster, sample_size) if len(cluster) > 5 else cluster
            
            for member_path in sample_members:
                # 获取基准图像特征
                member_features = feature_cache.get(member_path)
                if not member_features or 'sift' not in member_features:
                    continue
                    
                member_sift = member_features['sift']
                
                # 计算相似度
                try:
                    sim = self.sift_similarity(
                        member_path, candidate,
                        member_sift, candidate_sift
                    )
                except Exception as e:
                    print(f"相似度计算失败 {member_path} vs {candidate}: {str(e)}")
                    continue
                
                # 有效性检查
                if sim > 0:
                    total_sim += sim
                    valid_comparisons += 1
                    
                    # 提前终止条件：低匹配概率
                    if valid_comparisons >=3 and (total_sim/valid_comparisons) < (threshold*0.6):
                        break
            
            # 判断是否满足阈值
            if valid_comparisons > 0:
                avg_sim = total_sim / valid_comparisons
                if avg_sim >= threshold:
                    valid_candidates.append(candidate)
                    
        return valid_candidates



    def _precompute_features(self, img_path, progress_callback, total, index):
        try:
            if progress_callback:
                progress = int((index / total) * 30)
                progress_callback(progress)
                
            phash = self.pHash(img_path)
            hist = self._calc_histogram(img_path)
            return (img_path, {'phash': phash, 'hist': hist, 'sift': None})
        except Exception as e:
            raise RuntimeError(f"特征预计算失败 {img_path}: {str(e)}")
        


    def _coarse_compare(self, base_features, target_features, threshold):
        phash_sim = self.pHash_similarity(base_features['phash'], target_features['phash'])
        hist_sim = 1 - cv2.compareHist(base_features['hist'], target_features['hist'], cv2.HISTCMP_BHATTACHARYYA)
        
        # 阶段1：颜色特征验证
        if(hist_sim<0.75):
            return False
        
        # 阶段2：结构特征交叉验证
        phash_weight = 0.6 if phash_sim > 0.85 else 0.4
        combined_score = phash_weight * phash_sim + 0.6 * hist_sim
            
        return combined_score > threshold

    def _get_sift_features(self, img_path, feature_cache):
        features = feature_cache.get(img_path)
        if not isinstance(features, dict):
            print(f"警告: feature_cache[{img_path}] 的值不是字典，而是 {type(features)}")
            # 确保返回一个默认的字典结构
            features = {'phash': None, 'hist': None, 'sift': None}
        return features.get('sift')

    def _find_max_cluster(self, coarse_group, validation_matrix):
        visited = set()
        max_cluster = []
        for img in coarse_group:
            if img not in visited:
                cluster = []
                queue = [img]
                while queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.add(node)
                        cluster.append(node)
                        queue.extend([n for n in validation_matrix[node] if n not in visited])
                if len(cluster) > len(max_cluster):
                    max_cluster = cluster
        return max_cluster

    def _calculate_cluster_similarity(self, cluster, candidate, feature_cache):
        total_sim = 0.0
        valid_comparisons = 0
        
        candidate_sift = self._get_sift_features(candidate, feature_cache)
        if candidate_sift is None:
            return 0.0

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for member in cluster:
                member_sift = self._get_sift_features(member, feature_cache)
                futures.append(executor.submit(
                    self.sift_similarity, 
                    member, candidate, member_sift, candidate_sift
                ))
            
            for future in futures:
                sim = future.result()
                if sim > 0:
                    total_sim += sim
                    valid_comparisons += 1
        
        return total_sim / valid_comparisons if valid_comparisons > 0 else 0.0


    def pHash(self, image_path):
        """感知哈希算法"""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
                
            resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            dct = cv2.dct(np.float32(resized))
            roi = dct[0:16, 0:16]  # 取低频区域
            avg = np.mean(roi)
            return ''.join(['1' if x > avg else '0' for x in roi.flatten()])
        except Exception as e:
            print(f"pHash计算失败: {str(e)}")
            return None

    def _calc_histogram(self, img_path, bins=(8, 8, 8)):
        """计算HSV直方图"""
        try:
            img = cv2.imread(img_path)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
            return cv2.normalize(hist, hist).flatten()
        except Exception as e:
            print(f"直方图计算失败: {str(e)}")
            return None

    def sift_descriptor(self, img_path):
        """SIFT描述符提取"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"图像加载失败: {img_path}")
                return None 
            
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)  # 显式接收两个返回值
            return (kp, des) if des is not None else None  # 返回元组或None
        except Exception as e:
            print(f"SIFT特征提取失败: {str(e)}")
            return None

    @staticmethod
    def pHash_similarity(hash1, hash2):
        """汉明距离计算相似度"""
        if len(hash1) != len(hash2):
            return 0.0
        return 1 - sum(c1 != c2 for c1, c2 in zip(hash1, hash2)) / len(hash1)

    def sift_similarity(self, img_a, img_b, kp_des_a=None, kp_des_b=None, ransac_threshold=8.0):
        """SIFT相似度计算"""
        try:
            # 初始化SIFT检测器
            sift = cv2.SIFT_create()
            
            # 处理特征点描述符
            def get_features(img_path, kp_des):
                if kp_des is None:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"无法加载图像: {img_path}")
                    kp, des = sift.detectAndCompute(img, None)
                else:
                    kp, des = kp_des
                return kp, des

            # 获取关键点和描述符
            kp1, des1 = get_features(img_a, kp_des_a)
            kp2, des2 = get_features(img_b, kp_des_b)

            # 描述符有效性检查
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return 0.0

            # FLANN匹配器参数配置
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            # KNN匹配与Lowe's测试
            matches = flann.knnMatch(des1, des2, k=2)
            good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

            # 几何验证核心逻辑
            if len(good_matches) > 4:
                # 提取匹配点坐标
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
                
                # 计算单应性矩阵(RANSAC方法)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
                
                if M is not None and mask is not None:
                    # 计算内点比例
                    inlier_ratio = np.sum(mask) / len(mask)
                    # 计算匹配点基础相似度
                    match_score = len(good_matches)/max(len(des1), len(des2))
                    # 综合评分策略
                    return 0.6 * match_score + 0.4 * inlier_ratio
                else:
                    # 单应性计算失败时退回基础匹配
                    return len(good_matches)/max(len(des1), len(des2))
            else:
                # 匹配点不足时返回基础分
                return len(good_matches)/max(len(des1), len(des2))

        except cv2.error as e:
            print(f"OpenCV错误: {str(e)}")
            return 0.0
        except Exception as e:
            print(f"计算异常: {str(e)}")
            return 0.0