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

        # 新增双阈值控制
        self.coarse_threshold = 0.7  # 粗筛阈值默认值
        self.fine_threshold = 0.6     # 细筛阈值默认值

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
        #self.ui.compareButton.clicked.connect(self.ui_manager.toggle_compare_mode)
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

        # 添加算法选择下拉框
        self.ui.algorithmComboBox = QtWidgets.QComboBox(self.ui.controlGroup)
        self.ui.algorithmComboBox.addItems(["pHash", "SIFT", "Histogram", "Combined"])
        self.ui.verticalLayout_3.insertWidget(0, self.ui.algorithmComboBox)




        self.add_threshold_controls()

                # 添加触发检测按钮
        self.ui.detectButton = QtWidgets.QPushButton("开始检测", self.ui.controlGroup)
        self.ui.verticalLayout_3.addWidget(self.ui.detectButton)
        self.ui.detectButton.clicked.connect(self.start_similarity_detection)

        # 监听算法选择变化
        self.ui.algorithmComboBox.currentIndexChanged.connect(self.on_algorithm_changed)

            # 调用算法切换逻辑，确保初始化时显示正确的控件
        self.on_algorithm_changed()



    def on_algorithm_changed(self):
        """根据算法选择动态切换控件显示"""
        algorithm = self.ui.algorithmComboBox.currentText().lower()
        if algorithm == "combined":
            # 显示粗粒度和细粒度阈值控件
            self.coarseLabel.show()
            self.coarseSlider.show()
            self.fineLabel.show()
            self.fineSlider.show()
            # 隐藏单一阈值控件
            self.ui.thresholdLabel.hide()
            self.ui.thresholdSlider.hide()
        else:
            # 显示单一阈值控件
            self.ui.thresholdLabel.show()
            self.ui.thresholdSlider.show()
            # 隐藏粗粒度和细粒度阈值控件
            self.coarseLabel.hide()
            self.coarseSlider.hide()
            self.fineLabel.hide()
            self.fineSlider.hide()



    def add_threshold_controls(self):
        """新增双阈值控制面板"""
        # 粗粒度阈值
        coarse_layout = QtWidgets.QHBoxLayout()
        self.coarseLabel = QtWidgets.QLabel(f"粗筛阈值: {self.coarse_threshold:.2f}")
        self.coarseSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.coarseSlider.setRange(50, 95)  # 对应0.5-0.95
        self.coarseSlider.setValue(int(self.coarse_threshold*100))
        self.coarseSlider.valueChanged.connect(self.update_coarse_threshold)
        coarse_layout.addWidget(self.coarseLabel)
        coarse_layout.addWidget(self.coarseSlider)

        # 细粒度阈值 
        fine_layout = QtWidgets.QHBoxLayout()
        self.fineLabel = QtWidgets.QLabel(f"细筛阈值: {self.fine_threshold:.2f}")
        self.fineSlider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.fineSlider.setRange(30, 80)    # 对应0.3-0.8
        self.fineSlider.setValue(int(self.fine_threshold*100))
        self.fineSlider.valueChanged.connect(self.update_fine_threshold)
        fine_layout.addWidget(self.fineLabel)
        fine_layout.addWidget(self.fineSlider)

        # 添加到控制面板
        self.ui.verticalLayout_3.addLayout(coarse_layout)
        self.ui.verticalLayout_3.addLayout(fine_layout)

    def update_coarse_threshold(self, value):
        self.coarse_threshold = value / 100
        self.coarseLabel.setText(f"粗筛阈值: {self.coarse_threshold:.2f}")

    def update_fine_threshold(self, value):
        self.fine_threshold = value / 100  # 正确：将整数转换为浮点数
        self.fineLabel.setText(f"细筛阈值: {self.fine_threshold:.2f}")

    def start_similarity_detection(self):
        algorithm = self.ui.algorithmComboBox.currentText().lower()
        all_images = []
        for group_name, images in self.group_data.items():
            all_images.extend(images)
        
        # 根据算法类型调用不同检测方法
        if algorithm == "combined":
            # 两阶段检测模式
            similarity_results = self.similarity_algorithm.apply_combined_grouping(
                all_images,
                coarse_threshold=self.coarse_threshold,
                fine_threshold=self.fine_threshold
            )
        else:
            # 单算法模式（保持原有逻辑）
            similarity_results = self.similarity_algorithm.apply_single_algorithm_grouping(
                all_images, 
                threshold=self.ui_manager.similarity_threshold,
                algorithm=algorithm
            )
        
        # 统一更新分组数据
        self.group_data.clear()
        for group_name, images in similarity_results.items():
            self.group_data[group_name] = images
        
        # 带缩略图刷新的分组树更新
        self.group_manager.init_group_tree()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainApp()
    main_window.show()
    sys.exit(app.exec())