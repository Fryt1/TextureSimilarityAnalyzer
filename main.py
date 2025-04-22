from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QListWidgetItem
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QMenu
from Ui_ImageTool import Ui_MainWindow
from UI.ui_manager import UIManager
from UI.group_manager import GroupManager
from Core.similarity_algorithms import SimilarityAlgorithms
from UI.image_manager import ImageManager  # 引入新类
import os
import shutil
import sys


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()  # 实例化 UI 类
        self.ui.setupUi(self)  # 绑定 UI 到主窗口

        # 初始化分组数据
        self.group_data = {"未分类": []}

        # 创建 GroupManager 实例
        self.group_manager = GroupManager(self, self.ui, self.group_data)

        # 创建 SimilarityAlgorithms 实例
        self.similarity_algorithm = SimilarityAlgorithms(
            self.group_data,
            self.group_manager.add_group,
            self.group_manager.init_group_tree
        )

        # 创建 UIManager 实例，并传入 SimilarityAlgorithms
        self.ui_manager = UIManager(self, self.similarity_algorithm)

        # 创建 ImageManager 实例
        self.image_manager = ImageManager(self, self.group_data)  # 传递 MainApp 实例作为父窗口

        # 初始化界面
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        # 设置 similarityList 的右键菜单策略
        self.ui.similarityList.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        #self.ui.similarityList.customContextMenuRequested.connect(self.show_similarity_list_menu)
        
        self.ui.similarityList.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection  # 允许Shift/Ctrl多选
        )
        
        # 设置初始相似度阈值显示
        self.ui.thresholdLabel.setText(f"相似度阈值: {self.ui_manager.similarity_threshold:.2f}")

        # 设置 similarityList 的图标模式和大小
        self.ui.similarityList.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.ui.similarityList.setIconSize(QSize(100, 100))

        # 设置分组树右键菜单
        self.ui.groupTree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.groupTree.customContextMenuRequested.connect(self.group_manager.show_group_menu)
        self.ui.groupTree.itemClicked.connect(self.group_manager.on_group_clicked)

        # 连接信号与槽
        self.ui.thresholdSlider.valueChanged.connect(self.ui_manager.update_threshold)
        self.ui.compareButton.clicked.connect(self.ui_manager.toggle_compare_mode)
        self.ui.importButton.clicked.connect(self.image_manager.import_images)  # 调用 ImageManager 的方法
        #self.ui.similarityList.itemClicked.connect(self.display_preview)

        self.ui_manager.setup_preview_connection()  # 绑定预览功能
        self.image_manager.setup_context_menu()      # 绑定右键菜单

        # 添加操作菜单
        # 新连接方式
        self.ui.actionDelete.triggered.connect(self.image_manager.delete_selected_items)
        self.ui.actionRename.triggered.connect(self.image_manager.batch_rename)
        self.ui.actionMove.triggered.connect(self.image_manager.move_item)
        # 初始化分组树
        self.group_manager.init_group_tree()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec())