from PyQt6.QtWidgets import QFileDialog, QMessageBox, QListWidgetItem
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QInputDialog  

import os
import shutil


class ImageManager:
    def __init__(self, parent, group_data):
        self.parent = parent  
        self.group_data = group_data


    def setup_context_menu(self):
        self.parent.ui.similarityList.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.parent.ui.similarityList.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos):
        """显示批量操作菜单"""
        # 判断当前点击项是否在已选范围内
        clicked_item = self.parent.ui.similarityList.itemAt(pos)
        if clicked_item and clicked_item not in self.parent.ui.similarityList.selectedItems():
            self.parent.ui.similarityList.clearSelection()
            clicked_item.setSelected(True)
        
        # 获取最终选中项
        selected_items = self.parent.ui.similarityList.selectedItems()
        if not selected_items:
            return
        
        # 创建菜单
        menu = QMenu(self.parent.ui.similarityList)
        delete_action = menu.addAction(f"删除({len(selected_items)}项)")
        rename_action = menu.addAction(f"批量重命名({len(selected_items)}项)") 
        move_action = menu.addAction(f"移动({len(selected_items)}项)")
        
        # 绑定新方法
        delete_action.triggered.connect(self.delete_selected_items)
        rename_action.triggered.connect(self.batch_rename)
        move_action.triggered.connect(self.move_item)
        
        menu.exec(self.parent.ui.similarityList.viewport().mapToGlobal(pos))


    def import_images(self):
        """批量导入图片"""
        selected_files, _ = QFileDialog.getOpenFileNames(
            self.parent,  
            "选择图片文件",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if selected_files:
            self.group_data["未分类"].extend([os.path.abspath(f) for f in selected_files])
            self.parent.group_manager.init_group_tree()  # 触发分组树重建
    
    def delete_selected_items(self):
        """批量删除选中项"""
        selected_items = self.parent.ui.similarityList.selectedItems()
        if not selected_items:
            return

        # 统一确认对话框
        file_count = len(selected_items)
        reply = QMessageBox.question(
            self.parent,
            "批量删除确认",
            f"确定要永久删除选中的{file_count}个文件吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            removed_paths = []
            for item in selected_items:
                file_path = item.data(Qt.ItemDataRole.UserRole).get("path")
                try:
                    os.remove(file_path)
                    removed_paths.append(file_path)
                except Exception as e:
                    QMessageBox.critical(self.parent, "错误", f"删除失败: {str(e)}")
                    return
            
            # 批量更新分组数据
            for group in self.group_data.values():
                group[:] = [p for p in group if p not in removed_paths]
            
            # 批量删除列表项
            for item in selected_items:
                self.parent.ui.similarityList.takeItem(
                    self.parent.ui.similarityList.row(item)
                )
    def batch_rename(self):
        """批量模式化重命名"""
        selected_items = self.parent.ui.similarityList.selectedItems()
        if not selected_items:
            return
        
        
        base_name, ok = QInputDialog.getText(
            self.parent,  
            "输入命名模板",  
            "请输入文件命名模板（使用 %d 表示序号）：" 
        )
        if not ok or not base_name:
            return
        
        counter = 1
        for item in selected_items:
            old_path = item.data(Qt.ItemDataRole.UserRole).get("path")  # 改用UserRole路径
            dir_name = os.path.dirname(old_path)
            _, ext = os.path.splitext(old_path)
            
            new_name = base_name.replace("%d", str(counter))
            new_path = os.path.join(dir_name, f"{new_name}{ext}")
            
            try:
                os.rename(old_path, new_path)
                
                item.setText(os.path.basename(new_path))  # 仅显示文件名 
                item.setData(Qt.ItemDataRole.UserRole, {   # 更新存储路径
                    "path": new_path,
                    "group": item.data(Qt.ItemDataRole.UserRole).get("group")
                })
                self.parent.group_manager.update_group_paths(old_path, new_path)
                counter += 1
            except Exception as e:
                QMessageBox.critical(self.parent, "重命名失败", f"无法重命名文件：{e}")
            
    def move_item(self):
        """批量移动选中项"""
        selected_items = self.parent.ui.similarityList.selectedItems()
        if not selected_items:
            return
        
        folder = QFileDialog.getExistingDirectory(self.parent, "选择目标文件夹")
        if not folder:
            return
        
        moved_files = []
        for item in selected_items:
            old_path = item.text()
            file_name = os.path.basename(old_path)
            new_path = os.path.join(folder, file_name)
            
            try:
                shutil.move(old_path, new_path)
                item.setText(new_path)
                moved_files.append((old_path, new_path))
            except Exception as e:
                QMessageBox.critical(self.parent, "错误", f"移动失败: {str(e)}")
                return
        
        # 批量更新分组路径
        for old, new in moved_files:
            self.parent.group_manager.update_group_paths(old, new)