# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:33:42 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from uic.param import Ui_Form


class ParamUI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self, parent=None):
        super(ParamUI, self).__init__(parent)
        self.setupUi(self)
        self.ion_mode.addItems(['positive', 'negative'])
        self.match_method.addItems(['simple'])
        self.precursor_pos = ['[M+H]+', '[M+Na]+', '[M+K]+', '[M-H2O+H]+', '[M+H-NH3]+', '[M+NH4]+']
        self.precursor_neg = ['[M-H]-', '[M+Cl]-', '[M+Hac-H]-', '[M-H2O-H]-']
        self.add_precursor_type_box(self.listWidget_1, self.precursor_pos)
        self.add_precursor_type_box(self.listWidget_2, self.precursor_neg)
        
        
    def add_precursor_type_box(self, list_widget, precursor_list):
        for i in precursor_list:
            box = QtWidgets.QCheckBox(i)
            if i in ['[M+H]+', '[M-H]-']:
                box.setChecked(True)
            item = QtWidgets.QListWidgetItem()
            list_widget.addItem(item)
            list_widget.setItemWidget(item, box) 


    def get_chosen_precursor_type(self, list_widget):
        count = list_widget.count()
        cb_list = [list_widget.itemWidget(list_widget.item(i)) for i in range(count)]
        chooses = []
        for cb in cb_list:
            if cb.isChecked():
                chooses.append(cb.text())
        return chooses



if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = ParamUI()
    ui.show()
    sys.exit(app.exec_())