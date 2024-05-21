# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:22:39 2024

@author: DELL
"""

import os
import shutil
import numpy as np
import pandas as pd
from matchms.exporting import save_as_mgf

# from PyQt5.Qt import QThread
from PyQt5.QtCore import Qt, QVariant, QThread
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar

from uic import main
from uic.param_ui import ParamUI

from core import hpic
from core import matching
from core import peakeval
from core import formula
from core import tandem


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
        self.Initial()
        self.parameters = {'feature_extraction':{'ion_mode': 'positive', 'intensity_thres': 1000, 'snr_thres': 5.0, 'mass_inv': 1.0, 'rt_inv': 15.0},
                           'feature_matching':{'method': 'simple', 'rt_tol': 15, 'mz_tol': 0.01, 'min_frac': 0.5},
                           'precursor_type':{'positive': ['[M+H]+'], 'negative': ['[M-H]-']}}
        
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self.ParamUI = ParamUI()

        # action
        self.butt_open.clicked.connect(self.load_files)
        self.butt_run.clicked.connect(self.find_peaks)
        self.butt_param.clicked.connect(self.ParamUI.show)
        self.butt_save.clicked.connect(self.save_results)
        self.ParamUI.butt_ok.clicked.connect(self._set_parameters)
        self.ParamUI.butt_cancel.clicked.connect(self.ParamUI.close)
        self.list_files.itemClicked.connect(self.fill_peak_table)
        self.tab_peak.itemClicked.connect(self.plot_chromatogram)
        self.tab_feature.itemClicked.connect(self.plot_spectrum)
        
        # thread
        self.Thread_Feature = None
        self.Thread_Evaluating = None
        self.Thread_Matching = None
        self.Thread_Assigning = None
        
        # plot
        self.fig_spectrum = MakeFigure(3.6, 2.4, dpi = 300)
        self.fig_spectrum_ntb = NavigationToolbar(self.fig_spectrum, self)
        self.gridlayoutfigSpec = QGridLayout(self.box_spectrum)
        self.gridlayoutfigSpec.addWidget(self.fig_spectrum)
        self.gridlayoutfigSpec.addWidget(self.fig_spectrum_ntb)
        
        self.fig_chromatogram = MakeFigure(3.6, 2.4, dpi = 300)
        self.fig_chromatogram_ntb = NavigationToolbar(self.fig_chromatogram, self)
        self.gridlayoutfigChrom = QGridLayout(self.box_chrom)
        self.gridlayoutfigChrom.addWidget(self.fig_chromatogram)
        self.gridlayoutfigChrom.addWidget(self.fig_chromatogram_ntb)


    def Initial(self):
        self.files = []
        self.peaks = []
        self.peak_scores = []
        self.feature_table = None


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
        self.butt_param.setDisabled(True)
        

    def _set_finished(self):
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Ready')
        self.butt_open.setDisabled(False)
        self.butt_run.setDisabled(False)
        self.butt_save.setDisabled(False)
        self.butt_param.setDisabled(False)
        

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
        
        
    def _set_tracked_peaks(self, msg):
        self.peak_profile.append(msg)


    def _set_feature_table(self, msg):
        self.feature_table = msg
        # print(self.feature_table)


    def _set_parameters(self):
        self.parameters = {'feature_extraction':{'ion_mode': str(self.ParamUI.ion_mode.currentText()),
                                                 'intensity_thres': float(self.ParamUI.intensity_thres.value()),
                                                 'snr_thres': float(self.ParamUI.snr_thres.value()),
                                                 'mass_inv': float(self.ParamUI.mass_inv.value()),
                                                 'rt_inv': float(self.ParamUI.rt_inv.value())},
                           'feature_matching':{'method': str(self.ParamUI.match_method.currentText()),
                                               'rt_tol': float(self.ParamUI.match_rt_tol.value()),
                                               'mz_tol': float(self.ParamUI.match_mz_tol.value()),
                                               'min_frac': float(self.ParamUI.match_min_frac.value())},
                           'precursor_type':{'positive': self.ParamUI.get_chosen_precursor_type(self.ParamUI.listWidget_1),
                                             'negative': self.ParamUI.get_chosen_precursor_type(self.ParamUI.listWidget_2)}}
        self.ParamUI.close()
        print(self.parameters)


    def load_files(self):
        self.Initial()
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
        self.progressBar.setFormat('Extracting Features')
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
        self.progressBar.setFormat('Evaluating Features')
        self.Thread_Evaluating = Thread_Evaluating(self.files, self.peaks)
        self.Thread_Evaluating._i.connect(self._set_process_bar)
        self.Thread_Evaluating._result.connect(self._set_evaluated_peaks)
        self.Thread_Evaluating.start()
        self.Thread_Evaluating.finished.connect(self.assign_peak_score)
    

    def assign_peak_score(self):
        for i, f in enumerate(self.files):
            score = self.peak_scores[i]
            self.peaks[i]['peaks']['score'] = score
        self.match_peaks()
        

    def match_peaks(self):
        self.progressBar.setValue(40)
        self.progressBar.setFormat('Matching Features')
        self.Thread_Matching = Thread_Matching(peaks = self.peaks, 
                                               files = self.files,
                                               ion_mode = self.parameters['feature_extraction']['ion_mode'],
                                               method = self.parameters['feature_matching']['method'],
                                               rt_tol = self.parameters['feature_matching']['rt_tol'],
                                               mz_tol = self.parameters['feature_matching']['mz_tol'],
                                               min_frac = self.parameters['feature_matching']['min_frac'])
        self.Thread_Matching._result.connect(self._set_feature_table)
        self.Thread_Matching.start()
        self.Thread_Matching.finished.connect(self.assign_tandem_ms)


    def assign_tandem_ms(self):
        self.progressBar.setValue(70)
        self.progressBar.setFormat('Assigning MS/MS to Features')
        self.Thread_Assigning = Thread_Assigning(feature_table = self.feature_table, 
                                                 files = self.files, 
                                                 rt_tol = self.parameters['feature_matching']['rt_tol'],
                                                 mz_tol = self.parameters['feature_matching']['mz_tol'],
                                                 ion_mode = self.parameters['feature_extraction']['ion_mode'], 
                                                 precursor_type_list = self.parameters['precursor_type'][self.parameters['feature_extraction']['ion_mode']])
        self.Thread_Assigning._result.connect(self._set_feature_table)
        self.Thread_Assigning.start()
        self.Thread_Assigning.finished.connect(self.fill_feature_table)


    def fill_peak_table(self):
        selectItem = self.list_files.currentItem()
        if not selectItem:
            self.list_files.setCurrentRow(0)
        selectItem = selectItem.text()
        wh = self.files.index(selectItem)
        peak_table = self.peaks[wh]['peaks']
        self._set_table_widget(self.tab_peak, peak_table)


    def fill_feature_table(self):
        self.progressBar.setValue(100)
        self.progressBar.setFormat('Finished')
        feature_table = self.feature_table
        feature_table = feature_table.reset_index(drop = True)
        feature_table['Index'] = feature_table.index
        self._set_table_widget(self.tab_feature, feature_table)
        self.fill_peak_table()
        self._set_finished()


    def save_results(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        savePath = QtWidgets.QFileDialog.getExistingDirectory(self, "Save", options=options)
        if savePath:
            if savePath == '':
                self.WarnMsg('Invalid path')
                return
            feature_table = self.feature_table
            spectrum_list = [s for s in list(feature_table['Tandem_MS']) if s is not None]
            feature_table.to_csv(os.path.join(savePath, 'feature_table.csv'))
            save_as_mgf(spectrum_list, os.path.join(savePath, 'spectrum_list.mgf'))
        self.InforMsg('Finished')


    def plot_spectrum(self):
        wh = self.tab_feature.currentRow()
        wh = int(self.tab_feature.item(wh, 7).text())
        s = self.feature_table.loc[wh, 'Tandem_MS']
        self.fig_spectrum.PlotSpectrum(s)
        self.fill_information_table()
        
        
    def plot_chromatogram(self):
        wf = self.list_files.currentRow()
        wp = self.tab_peak.currentRow()
        rt1 = float(self.tab_peak.item(wp, 3).text())
        rt2 = float(self.tab_peak.item(wp, 4).text())
        pic_label = self.tab_peak.item(wp, 9).text()
        pic = self.peaks[wf]['pics'][pic_label]
        self.fig_chromatogram.PlotChromatogram(pic, rt1, rt2)


    def fill_information_table(self):
        wh = self.tab_feature.currentRow()
        wh = int(self.tab_feature.item(wh, 7).text())
        s = self.feature_table.loc[wh, 'Tandem_MS']
        information = s.metadata
        keys = [k for k in information.keys()]
        values = [information[k] for k in keys]
        info_table = pd.DataFrame({'keys':keys, 'values':values})
        self._set_table_widget(self.tab_info, info_table)
        
        

class Thread_Feature(QThread):
    _i = QtCore.pyqtSignal(int)
    _result = QtCore.pyqtSignal(dict)

    def __init__(self, files, intensity_thres, snr_thres, mass_inv, rt_inv):
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
    _result = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, peaks, files, ion_mode, method, rt_tol, mz_tol, min_frac):
        super(Thread_Matching, self).__init__()
        self.peaks = peaks
        self.files = files
        self.ion_mode = ion_mode
        self.method = method
        self.rt_tol = rt_tol
        self.mz_tol = mz_tol
        self.min_frac = min_frac

    def __del__(self):
        self.wait()
        self.working = False

    def run(self):
        linker = matching.FeatureMatching(self.peaks, self.files)
        if self.method == 'simple':
            feature_table = linker.simple_matching(mz_tol = self.mz_tol, rt_tol = self.rt_tol)
            feature_table = linker.feature_filter(min_frac = self.min_frac)
            feature_table['Ionmode'] = self.ion_mode
            self._result.emit(feature_table)


class Thread_Assigning(QThread):
    _result = QtCore.pyqtSignal(pd.DataFrame)

    def __init__(self, feature_table, files, rt_tol, mz_tol, ion_mode, precursor_type_list):
        super(Thread_Assigning, self).__init__()
        self.feature_table = feature_table
        self.files = files
        self.rt_tol = rt_tol
        self.mz_tol = mz_tol
        self.ion_mode = ion_mode
        self.precursor_type_list = precursor_type_list

    def __del__(self):
        self.wait()
        self.working = False

    def run(self):
        spectrums = tandem.load_tandem_ms(self.files)
        spectrums = tandem.cluster_tandem_ms(spectrums, mz_tol = self.mz_tol, rt_tol = self.rt_tol)
        feature_table = tandem.feature_spectrum_matching(self.feature_table, spectrums, mz_tol = self.mz_tol, rt_tol = self.rt_tol)
        feature_table = formula.assign_formula(feature_table, precursor_type_list = self.precursor_type_list, mz_tol = self.mz_tol)
        self._result.emit(feature_table)


class MakeFigure(FigureCanvas):
    def __init__(self,width=5, height=5, dpi=300):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.subplots_adjust(top=0.95,bottom=0.3,left=0.18,right=0.95)
        super(MakeFigure,self).__init__(self.fig) 
        self.axes = self.fig.add_subplot(111)
        self.axes.spines['bottom'].set_linewidth(0.5)
        self.axes.spines['left'].set_linewidth(0.5)
        self.axes.spines['right'].set_linewidth(0.5)
        self.axes.spines['top'].set_linewidth(0.5)
        self.axes.tick_params(width=0.8,labelsize=3)
        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        
        
    def PlotSpectrum(self, spectrum):
        self.axes.cla()
        mz, abunds = spectrum.peaks.mz, spectrum.peaks.intensities
        abunds /= np.max(abunds)
        self.axes.vlines(mz, ymin=0, ymax=abunds, color='r', lw = 0.5)
        self.axes.axhline(y=0,color='black', lw = 0.5)
        self.axes.set_xlabel('m/z', fontsize = 3.5)
        self.axes.set_ylabel('abundance', fontsize = 3.5)
        self.draw()
        
        
    def PlotChromatogram(self, pic, rtmin, rtmax):
        self.axes.cla()
        rt = pic[:,0]
        abunds = pic[:,2]
        self.axes.plot(rt, abunds)
        self.axes.axvspan(rtmin, rtmax, color='red', alpha=0.2)
        self.axes.set_xlabel('retention time', fontsize = 3.5)
        self.axes.set_ylabel('intensities', fontsize = 3.5)
        self.draw()



if __name__ == '__main__':
    import sys
    
    app = QApplication(sys.argv)
    ui = AutoMS()
    ui.show()
    sys.exit(app.exec_())
