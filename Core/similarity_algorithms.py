import random

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

    def apply_similarity_grouping(self, images, threshold):
        """
        根据相似度阈值对图片进行分组。

        :param images: 图片路径列表
        :param threshold: 相似度阈值
        """
        # 示例逻辑：将图片随机分组（你可以替换为实际的相似度算法）
        grouped_images = self.random_grouping(images, num_groups=5)

        # 清空现有分组
        self.group_data.clear()

        # 根据分组结果重新分组
        for group_name, paths in grouped_images.items():
            self.add_group_callback(group_name, paths)

        # 刷新分组树
        self.init_group_tree_callback()

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