from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QListWidgetItem
from PyQt6.QtCore import Qt
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize  # 确保导入 QSize
from Ui_ImageTool import Ui_MainWindow  # 导入生成的UI类
from PyQt6.QtWidgets import QTreeWidgetItem  # 新增导入
from PyQt6.QtWidgets import QMenu
import sys
import os
import shutil
import random


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  # 实例化UI类
        self.ui.setupUi(self)  # 绑定UI到主窗口

        self.ui.similarityList.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.MultiSelection)

        self.group_data = {
            "未分类": []  # 初始化默认分组
        }     

        # 初始化相似度阈值
        self.similarity_threshold = 0.85
        self.ui.thresholdLabel.setText(f"相似度阈值: {self.similarity_threshold:.2f}")


        self.ui.groupTree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.groupTree.customContextMenuRequested.connect(self.show_group_menu)

        # 连接信号与槽


        self.ui.thresholdSlider.valueChanged.connect(self.update_threshold)
        self.ui.compareButton.clicked.connect(lambda: self.toggle_compare_mode())
        self.ui.actionButton.clicked.connect(self.execute_action)
        self.ui.importButton.clicked.connect(self.import_images)
        self.ui.similarityList.itemClicked.connect(self.display_preview)

        # 创建操作菜单
        self.action_menu = QtWidgets.QMenu(self)
        self.action_menu.addAction(self.ui.actionDelete)
        self.action_menu.addAction(self.ui.actionRename)
        self.action_menu.addAction(self.ui.actionMove)
        self.ui.actionButton.setMenu(self.action_menu)  # 关键：绑定菜单到按钮

        # 绑定删除操作到 delete_selected_items 函数
        self.ui.actionDelete.triggered.connect(self.delete_selected_items)
        # 绑定重命名操作到 batch_rename 函数
        self.ui.actionRename.triggered.connect(self.batch_rename)
        # 绑定移动操作到 move_to_folder 函数
        self.ui.actionMove.triggered.connect(self.move_to_folder)

        self.ui.groupTree.itemClicked.connect(self.on_group_clicked)


        # 初始化界面
        self.init_ui()

    def init_ui(self):
        # 初始化格式过滤器选项

        # 设置 similarityList 的图标模式和大小
        self.ui.similarityList.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.ui.similarityList.setIconSize(QSize(100, 100))  # 设置图标大小


    def update_threshold(self, value):
        """更新相似度阈值"""
        self.similarity_threshold = value / 100
        self.ui.thresholdLabel.setText(f"相似度阈值: {self.similarity_threshold:.2f}")
        print(f"阈值更新: {self.similarity_threshold:.2f}")

    def toggle_compare_mode(self):
        """切换到对比模式"""

        QMessageBox.information(self, "提示", "已进入对比模式！")
         # 调用测试随机分组功能
        
        self.test_random_grouping()
        print("对比模式激活")

    def execute_action(self):
        """执行操作"""
        selected_items = self.ui.similarityList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择项目！")
            return

        for item in selected_items:
            print(f"操作项: {item.text()}")
        QMessageBox.information(self, "完成", "操作执行成功！")
    def import_images(self):
        """批量导入图片"""
        selected_files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择图片文件",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if selected_files:
            # 确保 "未分类" 分组存在且为列表
            if "未分类" not in self.group_data or not isinstance(self.group_data["未分类"], list):
                self.group_data["未分类"] = []

            # 将新图片添加到"未分类"分组
            self.group_data["未分类"].extend(selected_files)
            self.init_group_tree()  # 刷新分组显示
            
            # 高亮显示新添加的分组
            for i in range(self.ui.groupTree.topLevelItemCount()):
                item = self.ui.groupTree.topLevelItem(i)
                if item.data(0, Qt.ItemDataRole.UserRole)["name"] == "未分类":
                    self.ui.groupTree.setCurrentItem(item)
                    break

            # 更新 similarityList
            self.ui.similarityList.clear()
            self.ui.similarityList.setViewMode(QtWidgets.QListView.ViewMode.IconMode)  # 设置为图标模式
            self.ui.similarityList.setIconSize(QSize(100, 100))  # 设置图标大小

            for file_path in selected_files:
                pixmap = QPixmap(file_path).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                icon = QIcon(pixmap)
                item = QListWidgetItem(icon, file_path)
                self.ui.similarityList.addItem(item)


    def init_group_tree(self):
        """初始化带动态扩展能力的分组树"""
        self.ui.groupTree.clear()
        self.ui.groupTree.setHeaderLabel("贴图分组")
        
        # 动态生成所有分组节点
        for group_name, images in self.group_data.items():
            group_item = QTreeWidgetItem(self.ui.groupTree)
            group_item.setText(0, f"{group_name} ({len(images)})")
            
            # 设置分组缩略图（取组内首张图片）
            if images:
                pixmap = QPixmap(images[0]).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio)
                group_item.setIcon(0, QIcon(pixmap))
            
            # 存储完整数据
            group_item.setData(0, Qt.ItemDataRole.UserRole, {
                "name": group_name,
                "images": images.copy()
            })
        
        self.ui.groupTree.expandAll()


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
        """加载分组内容时同步更新数据"""
        group_info = item.data(0, Qt.ItemDataRole.UserRole)
        self.current_group = group_info["name"]
        
        self.ui.similarityList.clear()

        self.ui.similarityList.setViewMode(QtWidgets.QListView.ViewMode.IconMode)  # 设置为图标模式
        self.ui.similarityList.setIconSize(QSize(100, 100))  # 设置图标大小为 100x100
        for path in group_info["images"]:
            # 添加带状态检查的列表项
            list_item = QListWidgetItem(QIcon(QPixmap(path)), path)
            list_item.setData(Qt.ItemDataRole.UserRole, {
                "path": path,
                "group": self.current_group
            })
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
        menu = QMenu(self)
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
        new_name, ok = QtWidgets.QInputDialog.getText(self, "重命名分组", "输入新分组名称：", text=old_name)
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        if new_name in self.group_data:
            QMessageBox.warning(self, "警告", "分组名称已存在，请输入其他名称！")
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
            self,
            "删除分组",
            f"确定要删除分组 '{group_name}' 吗？分组内的图片不会被删除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 删除分组数据
            self.group_data.pop(group_name, None)
            self.ui.groupTree.takeTopLevelItem(self.ui.groupTree.indexOfTopLevelItem(item))

    def apply_similarity_grouping(self, threshold):
        """对 group_tree 中所有组的图片进行分类"""
        # 收集所有图片
        all_images = []
        for group_name, images in self.group_data.items():
            all_images.extend(images)

        # 假设 similarity_results 是相似度算法返回的分组结果
        # 这里需要替换为实际的相似度算法调用
        similarity_results = self.similarity_algorithm(all_images, threshold)

        # 清空现有分组
        self.group_data.clear()

        # 根据相似度结果重新分组
        for group_name, paths in similarity_results.items():
            unique_name = self.get_unique_groupname(group_name)
            self.add_group(unique_name, paths)

        # 刷新分组树
        self.init_group_tree()


    def random_grouping(self, images, num_groups=5):
        """
        将图片随机分成指定数量的组。
        
        :param images: 图片路径列表
        :param num_groups: 分组数量，默认为 5
        :return: 一个字典，键为组名，值为图片路径列表
        """
        # 初始化分组结果
        groups = {f"group_{i+1}": [] for i in range(num_groups)}

        # 随机分配图片到组
        for image in images:
            group_name = random.choice(list(groups.keys()))
            groups[group_name].append(image)

        return groups
    
    def test_random_grouping(self):
        """测试随机分组功能"""
        # 收集所有图片
        all_images = []
        for group_name, images in self.group_data.items():
            all_images.extend(images)

        # 调用随机分组函数
        random_groups = self.random_grouping(all_images, num_groups=5)

        # 清空现有分组
        self.group_data.clear()

        # 根据随机分组结果重新分组
        for group_name, paths in random_groups.items():
            self.add_group(group_name, paths)

        # 刷新分组树
        self.init_group_tree()


    def display_preview(self, item):
        """在 previewArea 中展示选中的图片"""
        # 获取图片路径
        image_path = item.text()

        # 加载图片并设置到 previewArea
        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            # 缩放图片以适应 previewArea 的大小
            scaled_pixmap = pixmap.scaled(
                self.ui.previewArea.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            # 创建 QGraphicsScene 并设置图片
            scene = QtWidgets.QGraphicsScene()
            scene.addPixmap(scaled_pixmap)

            # 将场景设置到 previewArea
            self.ui.previewArea.setScene(scene)
        else:
            QMessageBox.warning(self, "错误", f"无法加载图片: {image_path}")
            
    def delete_selected_items(self):
        """删除选中项"""
        selected_items = self.ui.similarityList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要删除的项目！")
            return

        for item in selected_items:
            # 获取项目在列表中的行号
            row = self.ui.similarityList.row(item)
            
            # 从 similarityList 中移除项目
            self.ui.similarityList.takeItem(row)
            
            # 获取图片文件路径（假设 item.text() 是文件名或完整路径）
            file_path = item.text()  # 如果是文件名，拼接完整路径，例如：os.path.join("images", item.text())
            
            # 删除本地文件
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"已删除本地文件: {file_path}")
                else:
                    print(f"文件不存在: {file_path}")
            except Exception as e:
                print(f"删除文件失败: {file_path}, 错误: {e}")
            
            # 打印删除的项目
            print(f"已删除: {item.text()}")

    def batch_rename(self):
        """批量重命名选中项"""
        selected_items = self.ui.similarityList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要重命名的项目！")
            return

        prefix, ok = QtWidgets.QInputDialog.getText(self, "批量重命名", "输入新名称前缀：")
        if not ok or not prefix.strip():
            return

        for index, item in enumerate(selected_items):
            old_path = item.text()
            dir_name = os.path.dirname(old_path)

            # 第一个文件没有后缀，后续文件从 1 开始编号
            if index == 0:
                new_name = f"{prefix}"
            else:
                new_name = f"{prefix}_{index}"

            # 保留原文件的扩展名
            new_name += os.path.splitext(old_path)[1]
            new_path = os.path.join(dir_name, new_name)

            os.rename(old_path, new_path)
            item.setText(new_path)
            print(f"已重命名: {old_path} -> {new_path}")

    def move_to_folder(self):
        """移动选中项到文件夹"""
        selected_items = self.ui.similarityList.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请先选择要移动的项目！")
            return

        folder = QFileDialog.getExistingDirectory(self, "选择目标文件夹")
        if not folder:
            return

        for item in selected_items:
            old_path = item.text()
            file_name = os.path.basename(old_path)
            new_path = os.path.join(folder, file_name)

            shutil.move(old_path, new_path)
            item.setText(new_path)
            print(f"已移动: {old_path} -> {new_path}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()

    sys.exit(app.exec())