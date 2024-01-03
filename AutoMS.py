# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:22:39 2024

@author: DELL
"""

import os
import re
import pickle
import shutil
import numpy as np
import pandas as pd

# from PyQt5.Qt import QThread
from PyQt5.QtCore import Qt, QVariant, QThread
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QLabel

from uic import main
from core import hpic
from core import peakeval


class AutoMS(QMainWindow, main.Ui_MainWindow):
    
    def __init__(self, parent=None):
        super(AutoMS, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("AutoMS")
        self.progressBar.setValue(0)
        
        try:
            shutil.rmtree('temp')  
            os.mkdir('temp') 
        except:
            pass
        
        # initial
        self.files = []
        self.peaks = []
        self.peak_scores = []
        self.ion_mode = 'positive'
        self.feature_table = None
        self.parameters = {'feature_extraction':{'intensity_thres': 1000, 'snr_thres': 5.0, 'mass_inv': 1.0, 'rt_inv': 15.0},
                           'feature_matching':{}}
        
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')

        # action
        self.butt_open.clicked.connect(self.load_files)
        self.butt_run.clicked.connect(self.find_peaks)
        
        self.list_files.itemClicked.connect(self.fill_peak_table)
        
        # thread
        self.Thread_Feature = None
        self.Thread_Evaluating = None


    def WarnMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Warning)
        msg.setText(Text)
        msg.setWindowTitle("Warning")
        msg.exec_()    
    
    
    def ErrorMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText(Text)
        msg.setWindowTitle("Error")
        msg.exec_()


    def InforMsg(self, Text):
        msg = QtWidgets.QMessageBox()
        msg.resize(550, 200)
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(Text)
        msg.setWindowTitle("Information")
        msg.exec_()


    def _set_busy(self):
        self.butt_open.setDisabled(True)
        self.butt_run.setDisabled(True)
        self.butt_save.setDisabled(True)
        

    def _set_finished(self):
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self.butt_open.setDisabled(False)
        self.butt_run.setDisabled(False)
        self.butt_save.setDisabled(False)
        

    def _set_table_widget(self, widget, data):
        widget.setRowCount(0)
        widget.setRowCount(data.shape[0])
        widget.setColumnCount(data.shape[1])
        widget.setHorizontalHeaderLabels(data.columns)
        widget.setVerticalHeaderLabels(data.index.astype(str))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if type(data.iloc[i,j]) == np.float64:
                    item = QtWidgets.QTableWidgetItem()
                    item.setData(Qt.EditRole, QVariant(float(data.iloc[i,j])))
                else:
                    item = QtWidgets.QTableWidgetItem(str(data.iloc[i,j]))
                widget.setItem(i, j, item)


    def _set_process_bar(self, msg):
        self.progressBar.setValue(int(msg))


    def _set_finished_peaks(self, msg):
        self.peaks.append(msg)
        
        
    def _set_evaluated_peaks(self, msg):
        self.peak_scores.append(msg)

    def _set_parameters(self):
        pass


    def load_files(self):
        self._set_busy()
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        fileNames, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Load", "","Data Files (*.mzml)", options=options)
        if len(fileNames) == 0:
            self._set_finished()
            return
        else:
            self.files = list(fileNames)
        self.set_list_files()
        self._set_finished()

        
    def set_list_files(self):
        data = self.files
        if len(data) == 0:
            return
        self.list_files.clear()
        for f in data:
            self.list_files.addItem(f)
        self.list_files.show()
        self.list_files.setCurrentRow(0)
        
    
    def find_peaks(self):
        self._set_busy()
        self.progressBar.setValue(0)
        self.progressBar.setFormat('Extracting Feature')
        self.peaks = []
        self.Thread_Feature = Thread_Feature(self.files,
                                             intensity_thres = self.parameters['feature_extraction']['intensity_thres'], 
                                             snr_thres = self.parameters['feature_extraction']['snr_thres'], 
                                             mass_inv = self.parameters['feature_extraction']['mass_inv'], 
                                             rt_inv = self.parameters['feature_extraction']['rt_inv'])
        self.Thread_Feature._result.connect(self._set_finished_peaks)
        self.Thread_Feature._i.connect(self._set_process_bar)
        self.Thread_Feature.start()
        self.Thread_Feature.finished.connect(self.evaluate_peaks)
        
        
    def evaluate_peaks(self):
        self.progressBar.setValue(0)
        self.progressBar.setFormat('Evaluating Feature')
        self.Thread_Evaluating = Thread_Evaluating(self.files, self.peaks)
        self.Thread_Evaluating._result.connect(self._set_evaluated_peaks)
        self.Thread_Evaluating._i.connect(self._set_process_bar)
        self.Thread_Evaluating.start()
        self.Thread_Evaluating.finished.connect(self.assign_peak_score)
    

    def assign_peak_score(self):
        for i, f in enumerate(self.files):
            score = self.peak_scores[i]
            self.peaks[i]['peaks']['score'] = score
        self.fill_peak_table()
    
    
    def fill_peak_table(self):
        selectItem = self.list_files.currentItem()
        if not selectItem:
            self.list_files.setCurrentRow(0)
        selectItem = selectItem.text()
        wh = self.files.index(selectItem)
        peak_table = self.peaks[wh]['peaks']
        self._set_table_widget(self.tab_peak, peak_table)




class Thread_Feature(QThread):
    _i = QtCore.pyqtSignal(int)
    _result = QtCore.pyqtSignal(dict)

    def __init__(self, files, intensity_thres = 1000, snr_thres = 5.0, mass_inv = 1.0, rt_inv = 15.0):
        super(Thread_Feature, self).__init__()
        self.files = files
        self.intensity_thres = intensity_thres
        self.snr_thres = snr_thres
        self.mass_inv = mass_inv
        self.rt_inv = rt_inv

    def __del__(self):
        self.wait()
        self.working = False

    def run(self):
        for i, f in enumerate(self.files):
            peaks, pics = hpic.hpic(f, 
                                    min_intensity = self.intensity_thres, 
                                    min_snr = self.snr_thres, 
                                    mass_inv = self.mass_inv, 
                                    rt_inv = self.rt_inv,
                                    max_items = 30000)
            self._i.emit(int(100 * (1+i) / len(self.files)))
            self._result.emit({'peaks': peaks, 'pics': pics})


class Thread_Evaluating(QThread):
    _i = QtCore.pyqtSignal(int)
    _result = QtCore.pyqtSignal(list)

    def __init__(self, files, peaks):
        super(Thread_Evaluating, self).__init__()
        self.peaks = peaks
        self.files = files

    def __del__(self):
        self.wait()
        self.working = False

    def run(self):
        for i, f in enumerate(self.files):
            vals = self.peaks[i]
            peak = vals['peaks']
            pic = vals['pics']
            score = peakeval.evaluate_peaks(peak, pic)
            self._i.emit(int(100 * (1+i) / len(self.files)))
            self._result.emit(list(score))


class Thread_Matching(QThread):
    _i = QtCore.pyqtSignal(int)
    _result = QtCore.pyqtSignal(dict)

    def __init__(self):
        super(Thread_Matching, self).__init__()
        pass

    def __del__(self):
        self.wait()
        self.working = False

    def run(self):
        pass





if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = AutoMS()
    ui.show()
    sys.exit(app.exec_())






