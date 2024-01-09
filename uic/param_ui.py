# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:33:42 2024

@author: DELL
"""

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from .param import Ui_Form


class ParamUI(QtWidgets.QWidget, Ui_Form):
    
    def __init__(self, parent=None):
        super(ParamUI, self).__init__(parent)
        self.setupUi(self)
        self.ion_mode.addItems(['Positive', 'Negative'])
        self.match_method.addItems(['Simple'])



if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = ParamUI()
    ui.show()
    sys.exit(app.exec_())