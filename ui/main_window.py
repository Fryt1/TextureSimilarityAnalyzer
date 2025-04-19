# ui/main_window.py
from PyQt6.QtCore import Qt, QThreadPool
from PyQt6.QtWidgets import QMainWindow, QFileDialog
from PyQt6.uic import loadUi
from core.feature_db import FeatureDatabase
from .worker import FeatureExtractor, SimilarityCalculator

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi('Ui_ImageTool.py', self)
        
        # 初始化组件
        self.feature_db = FeatureDatabase()
        self.threadpool = QThreadPool()
        self.current_group = None
        
        # 连接信号
        self.thresholdSlider.valueChanged.connect(self.update_threshold)
        self.actionButton.clicked.connect(self.handle_actions)
        self.groupTree.itemClicked.connect(self.on_group_selected)
        
    def load_images(self):
        """加载图片并提取特征"""
        path = QFileDialog.getExistingDirectory()
        if not path: return
        
        # 创建后台任务
        extractor = FeatureExtractor(path)
        extractor.signals.progress.connect(self.statusbar.showMessage)
        extractor.signals.finished.connect(self.on_features_loaded)
        self.threadpool.start(extractor)
    
    def on_features_loaded(self, features):
        """特征加载完成回调"""
        self.feature_db.update(features)
        self.run_similarity_calculation()
    
    def run_similarity_calculation(self):
        """启动相似度计算"""
        calculator = SimilarityCalculator(self.feature_db)
        calculator.signals.progress.connect(self.update_progress)
        calculator.signals.result.connect(self.display_groups)
        self.threadpool.start(calculator)
    
    def update_threshold(self, value):
        """阈值更新处理"""
        threshold = value / 100.0
        self.thresholdLabel.setText(f"阈值: {threshold:.2f}")
        self.feature_db.threshold = threshold
        self.run_similarity_calculation()
    
    def display_groups(self, groups):
        """显示分组结果"""
        self.groupTree.clear()
        for group_id, items in groups.items():
            parent = QTreeWidgetItem([f"Group {group_id} ({len(items)} images)"])
            for img_path in items:
                child = QTreeWidgetItem([os.path.basename(img_path)])
                parent.addChild(child)
            self.groupTree.addTopLevelItem(parent)