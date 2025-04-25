from PyQt6.QtWidgets import QTreeWidgetItem, QMenu, QMessageBox, QListWidgetItem
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt, QSize, QObject, QRunnable, QThreadPool, pyqtSignal
from PyQt6 import QtWidgets
import os

class GroupManager:
    def __init__(self, parent, ui, group_data):
        self.parent = parent
        self.ui = ui
        self.group_data = group_data
        self.current_group = None
        self.thread_pool = QThreadPool.globalInstance()  # 统一线程池管理



    def show_group_menu(self, pos):
        """显示分组右键菜单[6,7](@ref)"""
        item = self.ui.groupTree.itemAt(pos)
        if not item:
            return

        # 创建菜单
        menu = QMenu(self.ui.groupTree)
        rename_action = menu.addAction("重命名分组")
        delete_action = menu.addAction("删除分组")

        # 绑定操作
        action = menu.exec(self.ui.groupTree.viewport().mapToGlobal(pos))
        if action == rename_action:
            self.rename_group(item)
        elif action == delete_action:
            self.delete_group(item)

    class ThumbnailSignals(QObject):
        thumbnail_ready = pyqtSignal(QTreeWidgetItem, QPixmap)  # 增加item参数

    class ThumbnailWorker(QRunnable):
        def __init__(self, path, item):
            super().__init__()
            self.path = path
            self.item = item
            self.signals = GroupManager.ThumbnailSignals()

        def run(self):
            try:
                pixmap = QPixmap(self.path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(100, 100, 
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation)
                    self.signals.thumbnail_ready.emit(self.item, pixmap)
            except Exception as e:
                print(f"缩略图加载失败: {str(e)}")

    def init_group_tree(self):
        self.ui.groupTree.clear()
        self.ui.groupTree.setHeaderLabel("贴图分组")

        for group_name, images in self.group_data.items():
            group_item = QTreeWidgetItem(self.ui.groupTree)
            group_item.setText(0, f"{group_name} ({len(images)})")
            group_item.setData(0, Qt.ItemDataRole.UserRole, {
                "name": group_name,
                "images": [os.path.abspath(img) for img in images]
            })

            if images:
                worker = self.ThumbnailWorker(images[0], group_item)
                worker.signals.thumbnail_ready.connect(self._update_thumbnail)
                self.thread_pool.start(worker)

    def _update_thumbnail(self, item, pixmap):
        """线程安全的缩略图更新"""
        if item and not pixmap.isNull():
            item.setIcon(0, QIcon(pixmap))

    def add_group(self, group_name, images):
        if group_name not in self.group_data:
            with self.parent.data_lock:  # 使用数据锁
                self.group_data[group_name] = images
            self.init_group_tree()

    def move_to_group(self, target_group, image_paths):
        with self.parent.data_lock:  # 加锁操作共享数据
            for path in image_paths:
                # 从原分组移除
                for group in self.group_data.values():
                    if path in group:
                        group.remove(path)
                # 添加到新分组
                self.group_data[target_group].append(path)
        self._refresh_group_tree()

    def _refresh_group_tree(self):
        """局部刷新替代完全重建"""
        current_selected = self.ui.groupTree.currentItem()
        self.init_group_tree()
        if current_selected:
            self.ui.groupTree.setCurrentItem(current_selected)

    def on_group_clicked(self, item):
        if not (group_info := item.data(0, Qt.ItemDataRole.UserRole)):
            QMessageBox.warning(self.ui.groupTree, "错误", "分组数据异常")
            return

        self.current_group = group_info["name"]
        valid_images = [img for img in group_info["images"] if os.path.exists(img)]
        
        self.ui.similarityList.clear()
        for path in valid_images:
            list_item = QListWidgetItem()
            list_item.setIcon(QIcon(QPixmap(path).scaled(100, 100)))
            list_item.setData(Qt.ItemDataRole.UserRole, {
                "path": path,
                "group": self.current_group
            })
            self.ui.similarityList.addItem(list_item)

    # 其他方法保持原有逻辑，但需要添加数据锁保护
    def apply_similarity_grouping(self, threshold, similarity_algorithm):
        with self.parent.data_lock:
            all_images = [img for group in self.group_data.values() for img in group]
            similarity_results = similarity_algorithm(all_images, threshold)
            
            self.group_data.clear()
            for group_name, paths in similarity_results.items():
                self.add_group(self.get_unique_groupname(group_name), paths)

    def update_group_paths(self, old_path, new_path):
        with self.parent.data_lock:
            # 更新分组数据
            for group in self.group_data.values():
                if old_path in group:
                    group[group.index(old_path)] = new_path
            
            # 更新树形控件
            root = self.ui.groupTree.invisibleRootItem()
            for i in range(root.childCount()):
                item = root.child(i)
                if (group_info := item.data(0, Qt.ItemDataRole.UserRole)) and old_path in group_info["images"]:
                    group_info["images"][group_info["images"].index(old_path)] = new_path

            # 更新列表控件
            for i in range(self.parent.ui.similarityList.count()):
                if (item := self.parent.ui.similarityList.item(i)) and \
                   item.data(Qt.ItemDataRole.UserRole)["path"] == old_path:
                    item.setData(Qt.ItemDataRole.UserRole, {
                        "path": new_path,
                        "group": item.data(Qt.ItemDataRole.UserRole)["group"]
                    })