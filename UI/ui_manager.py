from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QListWidgetItem
from PyQt6.QtWidgets import QTreeWidgetItem, QMenu, QGraphicsScene  # 确保导入 QGraphicsScene
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
import os
class UIManager:
    def __init__(self, main_app, similarity_algorithm):
        self.main_app = main_app
        self.similarity_algorithm = similarity_algorithm
        self.similarity_threshold = 0.85

    def setup_preview_connection(self):
        self.main_app.ui.similarityList.itemClicked.connect(self.display_preview)
    
    def display_preview(self, item):
        # 从 UserRole 获取路径
        image_path = item.data(Qt.ItemDataRole.UserRole).get("path")
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self.main_app, "错误", "无法加载图片，文件可能已被移动或删除！")
            return

        pixmap = QPixmap(image_path)
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.main_app.ui.previewArea.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            scene = QGraphicsScene()
            scene.addPixmap(scaled_pixmap)
            self.main_app.ui.previewArea.setScene(scene)
        else:
            QMessageBox.warning(self.main_app, "错误", "无法加载图片！")

    def create_action_menu(self, parent):
        menu = QMenu(parent)
        menu.addAction(parent.ui.actionDelete)
        menu.addAction(parent.ui.actionRename) 
        menu.addAction(parent.ui.moveAction)
        return menu

    def update_threshold(self, value):
        self.similarity_threshold = value / 100
        self.main_app.ui.thresholdLabel.setText(f"相似度阈值: {self.similarity_threshold:.2f}")
        print(f"阈值更新: {self.similarity_threshold:.2f}")

    def toggle_compare_mode(self):
        QMessageBox.information(self.main_app, "提示", "已进入对比模式！")
        all_images = []
        for group_name, images in self.main_app.group_data.items():
            all_images.extend(images)
        self.similarity_algorithm.apply_similarity_grouping(
            images=all_images,
            threshold=self.similarity_threshold
        )
        print("对比模式激活并完成分组")  # 移除多余文本