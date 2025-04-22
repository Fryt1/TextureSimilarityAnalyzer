from PyQt6.QtWidgets import QTreeWidgetItem, QMenu, QMessageBox, QListWidgetItem
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize
from PyQt6 import QtWidgets

import os

class GroupManager:
    def __init__(self,parent, ui, group_data):
        """
        初始化 GroupManager 类。

        :param ui: 主窗口的 UI 对象
        :param group_data: 分组数据（字典）
        """
        self.parent = parent
        self.ui = ui
        self.group_data = group_data
        self.current_group = None


    

    def init_group_tree(self):
        """初始化带动态扩展能力的分组树（修复版）"""
        self.ui.groupTree.clear()
        self.ui.groupTree.setHeaderLabel("贴图分组")
        
        # 显式设置列数（网页2、网页4）
        self.ui.groupTree.setColumnCount(1)
        
        # 动态生成所有分组节点
        for group_name, images in self.group_data.items():
            group_item = QTreeWidgetItem(self.ui.groupTree)
            
            # 设置分组显示文本（带数量统计）
            group_item.setText(0, f"{group_name} ({len(images)})")
            
            # 设置缩略图（网页3、网页5）
            if images:
                try:
                    # 使用绝对路径并验证文件存在性（网页7）
                    valid_images = [os.path.abspath(img) for img in images if os.path.exists(img)]
                    if valid_images:
                        pixmap = QPixmap(valid_images[0])
                        if not pixmap.isNull():
                            pixmap = pixmap.scaled(100, 100, 
                                Qt.AspectRatioMode.KeepAspectRatio,
                                Qt.TransformationMode.SmoothTransformation)
                            group_item.setIcon(0, QIcon(pixmap))
                except Exception as e:
                    print(f"缩略图加载失败: {e}")
            
            # 存储完整分组数据（网页6）
            group_item.setData(0, Qt.ItemDataRole.UserRole, {
                "name": group_name,
                "images": [os.path.abspath(img) for img in images]  # 存储绝对路径
            })
        
        # 绑定点击信号（网页1、网页5）
        self.ui.groupTree.itemClicked.connect(self.on_group_clicked)
        
        # 展开所有节点（网页4）
        self.ui.groupTree.expandAll()
        
        # 强制刷新界面（网页7）
        self.ui.groupTree.viewport().update()

    def add_group(self, group_name, images):
        """添加新分组"""
        if group_name not in self.group_data:
            self.group_data[group_name] = images
            self.init_group_tree()  # 刷新树形结构

    def move_to_group(self, target_group, image_paths):
        """移动图片到指定分组"""
        for path in image_paths:
            # 从原分组移除
            for group in self.group_data.values():
                if path in group:
                    group.remove(path)
            # 添加到新分组
            self.group_data[target_group].append(path)
        self.init_group_tree()

    def on_group_clicked(self, item):
        group_info = item.data(0, Qt.ItemDataRole.UserRole)
        if not group_info or "images" not in group_info:
            QMessageBox.warning(self.ui.groupTree, "错误", "分组数据异常")
            return

        self.current_group = group_info["name"]
        images = [img for img in group_info["images"] if os.path.exists(img)]  # 过滤无效路径
        
        self.ui.similarityList.clear()
        self.ui.similarityList.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.ui.similarityList.setIconSize(QSize(100, 100))
        
        for path in images:
            if QPixmap(path).isNull():  # 验证图片可加载性
                print(f"无法加载图片: {path}")
                continue
            list_item = QListWidgetItem(QIcon(QPixmap(path)), path)
            list_item.setData(Qt.ItemDataRole.UserRole, {"path": path, "group": self.current_group})
            self.ui.similarityList.addItem(list_item)

    def get_unique_groupname(self, base_name):
        """生成唯一分组名"""
        counter = 1
        new_name = base_name
        while new_name in self.group_data:
            new_name = f"{base_name}_{counter}"
            counter += 1
        return new_name

    def show_group_menu(self, pos):
        """显示分组右键菜单"""
        # 获取当前点击的分组项
        item = self.ui.groupTree.itemAt(pos)
        if not item:
            return

        # 创建右键菜单
        menu = QMenu(self.ui.groupTree)
        rename_action = menu.addAction("重命名分组")
        delete_action = menu.addAction("删除分组")

        # 绑定操作
        action = menu.exec(self.ui.groupTree.viewport().mapToGlobal(pos))
        if action == rename_action:
            self.rename_group(item)
        elif action == delete_action:
            self.delete_group(item)

    def rename_group(self, item):
        """重命名分组"""
        group_info = item.data(0, Qt.ItemDataRole.UserRole)
        old_name = group_info["name"]

        # 弹出输入框获取新名称
        new_name, ok = QtWidgets.QInputDialog.getText(self.ui.groupTree, "重命名分组", "输入新分组名称：", text=old_name)
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        if new_name in self.group_data:
            QMessageBox.warning(self.ui.groupTree, "警告", "分组名称已存在，请输入其他名称！")
            return

        # 更新分组数据
        self.group_data[new_name] = self.group_data.pop(old_name)
        group_info["name"] = new_name
        item.setText(0, f"{new_name} ({len(group_info['images'])})")
        item.setData(0, Qt.ItemDataRole.UserRole, group_info)

    def delete_group(self, item):
        """删除分组"""
        group_info = item.data(0, Qt.ItemDataRole.UserRole)
        group_name = group_info["name"]

        # 确认删除
        reply = QMessageBox.question(
            self.ui.groupTree,
            "删除分组",
            f"确定要删除分组 '{group_name}' 吗？分组内的图片不会被删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 删除分组数据
            self.group_data.pop(group_name, None)
            self.ui.groupTree.takeTopLevelItem(self.ui.groupTree.indexOfTopLevelItem(item))

    def apply_similarity_grouping(self, threshold, similarity_algorithm):
        """
        对 group_tree 中所有组的图片进行分类。

        :param threshold: 相似度阈值
        :param similarity_algorithm: 相似度算法函数
        """
        # 收集所有图片
        all_images = []
        for group_name, images in self.group_data.items():
            all_images.extend(images)

        # 调用相似度算法获取分组结果
        similarity_results = similarity_algorithm(all_images, threshold)

        # 清空现有分组
        self.group_data.clear()

        # 根据相似度结果重新分组
        for group_name, paths in similarity_results.items():
            unique_name = self.get_unique_groupname(group_name)
            self.add_group(unique_name, paths)

        # 刷新分组树
        self.init_group_tree()

    def update_group_paths(self, old_path, new_path):
        """递归更新所有数据结构中的路径"""
        # 更新分组数据
        for group in self.group_data.values():
            if old_path in group:
                group[group.index(old_path)] = new_path
        
        # 更新树形控件数据
        root = self.ui.groupTree.invisibleRootItem()
        for i in range(root.childCount()):
            group_item = root.child(i)
            group_info = group_item.data(0, Qt.ItemDataRole.UserRole)
            if old_path in group_info["images"]:
                group_info["images"][group_info["images"].index(old_path)] = new_path
        
        # 更新列表控件数据（新增）
        for i in range(self.parent.ui.similarityList.count()):
            item = self.parent.ui.similarityList.item(i)
            if item.data(Qt.ItemDataRole.UserRole)["path"] == old_path:
                item.setData( Qt.ItemDataRole.UserRole, {"path": new_path, "group": ...})