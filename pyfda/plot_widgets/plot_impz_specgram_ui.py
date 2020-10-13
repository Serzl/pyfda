# -*- coding: utf-8 -*-
#
# This file is part of the pyFDA project hosted at https://github.com/chipmuenk/pyfda
#
# Copyright © pyFDA Project Contributors
# Licensed under the terms of the MIT License
# (see file LICENSE in root directory for details)

"""
Create the UI for the PlotImz class
"""
import logging
logger = logging.getLogger(__name__)

from pyfda.libs.compat import (QCheckBox, QWidget, QComboBox, QLineEdit, QLabel, QPushButton,
                      QHBoxLayout, QVBoxLayout, pyqtSignal, QEvent, Qt)

import numpy as np
from pyfda.libs.pyfda_lib import to_html, safe_eval
import pyfda.filterbroker as fb
from pyfda.libs.pyfda_qt_lib import qget_cmb_box, qset_cmb_box, qstyle_widget
from pyfda.libs.pyfda_fft_windows_lib import get_window_names, calc_window_function
from .plot_fft_win import Plot_FFT_win
from pyfda.pyfda_rc import params # FMT string for QLineEdit fields, e.g. '{:.3g}'

class PlotImpzSpecgram_UI(QWidget):
    """
    Create the UI for the spectrogram widget
    """
    # incoming: not implemented at the moment, update_N is triggered directly
    # by plot_impz
    # sig_rx = pyqtSignal(object)
    # outgoing: from various UI elements to PlotImpz ('ui_changed':'xxx')
    sig_tx = pyqtSignal(object)
    # outgoing to local fft window
    sig_tx_fft = pyqtSignal(object)


    def __init__(self, parent):
        """
        Pass instance `parent` of parent class (PlotImpz)
        """
        super(PlotImpz_UI, self).__init__(parent)

        """
        Intitialize the widget, consisting of:
        - top chkbox row
        - coefficient table
        - two bottom rows with action buttons
        """

        # initial settings
        self.N_start = 0
        self.N_user = 0
        self.N = 0

        self.bottom_f = -120 # initial value for log. scale
        self.param = None

        # initial settings for comboboxes
        self.plt_spec_resp = "Line"
        self.plt_spec_stim = "None"
        self.plt_spec_stmq = "None"

        # dictionary for fft window settings
        self.win_dict = fb.fil[0]['win_fft']
        self.fft_window = None # handle for FFT window pop-up widget
        self.window_name = "Rectangular"

        self._construct_UI()
        self._enable_stim_widgets()
        self.update_N(emit=False) # also updates window function
        self._update_noi()


    def _construct_UI(self):

        # ---------------------------------------------------------------
        # Controls for frequency domain
        # ---------------------------------------------------------------
        lbl_plt_spec_title = QLabel("<b>View:</b>", self)
        lbl_plt_spec_resp = QLabel(to_html("Response Y", frmt='bi'), self)
        self.cmb_plt_spec_resp = QComboBox(self)
        self.cmb_plt_spec_resp.addItems(plot_styles_list)
        qset_cmb_box(self.cmb_plt_spec_resp, self.plt_spec_resp)
        self.cmb_plt_spec_resp.setToolTip("<span>Plot style for response.</span>")

        self.lbl_plt_spec_stim = QLabel(to_html("Stimulus X", frmt='bi'), self)
        self.cmb_plt_spec_stim = QComboBox(self)
        self.cmb_plt_spec_stim.addItems(plot_styles_list)
        qset_cmb_box(self.cmb_plt_spec_stim, self.plt_spec_stim)
        self.cmb_plt_spec_stim.setToolTip("<span>Plot style for stimulus.</span>")

        self.lbl_plt_spec_stmq = QLabel("Fixp. Stim. " + to_html("X_Q", frmt='bi'), self)
        self.cmb_plt_spec_stmq = QComboBox(self)
        self.cmb_plt_spec_stmq.addItems(plot_styles_list)
        qset_cmb_box(self.cmb_plt_spec_stmq, self.plt_spec_stmq)
        self.cmb_plt_spec_stmq.setToolTip("<span>Plot style for <em>fixpoint</em> (quantized) stimulus.</span>")

        self.chk_log_spec = QCheckBox("dB : min.", self)
        self.chk_log_spec.setObjectName("chk_log_spec")
        self.chk_log_spec.setToolTip("<span>Logarithmic scale for y-axis.</span>")
        self.chk_log_spec.setChecked(True)

        self.led_log_bottom_spec = QLineEdit(self)
        self.led_log_bottom_spec.setText(str(self.bottom_f))
        self.led_log_bottom_spec.setToolTip("<span>Minimum display value for log. scale.</span>")
        self.led_log_bottom_spec.setVisible(self.chk_log_spec.isChecked())

        if not self.chk_log_spec.isChecked():
            self.chk_log_spec.setText("dB")
            self.bottom_f = 0

        self.lbl_win_fft = QLabel("Window: ", self)
        self.cmb_win_fft = QComboBox(self)
        self.cmb_win_fft.addItems(get_window_names())
        self.cmb_win_fft.setToolTip("FFT window type.")
        qset_cmb_box(self.cmb_win_fft, self.window_name)

        self.cmb_win_fft_variant = QComboBox(self)
        self.cmb_win_fft_variant.setToolTip("FFT window variant.")
        self.cmb_win_fft_variant.setVisible(False)

        self.lblWinPar1 = QLabel("Param1")
        self.ledWinPar1 = QLineEdit(self)
        self.ledWinPar1.setText("1")
        self.ledWinPar1.setObjectName("ledWinPar1")

        self.lblWinPar2 = QLabel("Param2")
        self.ledWinPar2 = QLineEdit(self)
        self.ledWinPar2.setText("2")
        self.ledWinPar2.setObjectName("ledWinPar2")

        self.chk_Hf = QCheckBox(self)
        self.chk_Hf.setObjectName("chk_Hf")
        self.chk_Hf.setToolTip("<span>Show ideal frequency response, calculated "
                               "from the filter coefficients.</span>")
        self.chk_Hf.setChecked(False)
        self.chk_Hf_lbl = QLabel(to_html("H_id (f)", frmt="bi"), self)

        layH_ctrl_spec = QHBoxLayout()
        layH_ctrl_spec.addWidget(lbl_plt_spec_title)
        layH_ctrl_spec.addStretch(1)
        layH_ctrl_spec.addWidget(lbl_plt_spec_resp)
        layH_ctrl_spec.addWidget(self.cmb_plt_spec_resp)
        layH_ctrl_spec.addStretch(1)
        layH_ctrl_spec.addWidget(self.lbl_plt_spec_stim)
        layH_ctrl_spec.addWidget(self.cmb_plt_spec_stim)
        layH_ctrl_spec.addStretch(1)
        layH_ctrl_spec.addWidget(self.lbl_plt_spec_stmq)
        layH_ctrl_spec.addWidget(self.cmb_plt_spec_stmq)
        layH_ctrl_spec.addStretch(2)
        layH_ctrl_spec.addWidget(self.chk_log_spec)
        layH_ctrl_spec.addWidget(self.led_log_bottom_spec)
        layH_ctrl_spec.addStretch(2)
        layH_ctrl_spec.addWidget(self.lbl_win_fft)
        layH_ctrl_spec.addWidget(self.cmb_win_fft)
        layH_ctrl_spec.addWidget(self.cmb_win_fft_variant)
        layH_ctrl_spec.addWidget(self.lblWinPar1)
        layH_ctrl_spec.addWidget(self.ledWinPar1)
        layH_ctrl_spec.addWidget(self.lblWinPar2)
        layH_ctrl_spec.addWidget(self.ledWinPar2)
        layH_ctrl_spec.addStretch(2)
        layH_ctrl_spec.addWidget(self.chk_Hf)
        layH_ctrl_spec.addWidget(self.chk_Hf_lbl)
        layH_ctrl_spec.addStretch(10)

        #layH_ctrl_spec.setContentsMargins(*params['wdg_margins'])

        self.wdg_ctrl_spec = QWidget(self)
        self.wdg_ctrl_spec.setLayout(layH_ctrl_spec)
        # ---- end Frequency Domain ------------------


        #----------------------------------------------------------------------
        # LOCAL SIGNALS & SLOTs
        #----------------------------------------------------------------------

        # --- frequency control ---
        # careful! currentIndexChanged passes the current index to _update_win_fft
        self.cmb_win_fft.currentIndexChanged.connect(self._update_win_fft)
        self.ledWinPar1.editingFinished.connect(self._read_param1)
        self.ledWinPar2.editingFinished.connect(self._read_param2)


#------------------------------------------------------------------------------
    def eventFilter(self, source, event):
        """
        Filter all events generated by the monitored widgets. Source and type
        of all events generated by monitored objects are passed to this eventFilter,
        evaluated and passed on to the next hierarchy level.

        - When a QLineEdit widget gains input focus (``QEvent.FocusIn``), display
          the stored value from filter dict with full precision
        - When a key is pressed inside the text field, set the `spec_edited` flag
          to True.
        - When a QLineEdit widget loses input focus (``QEvent.FocusOut``), store
          current value normalized to f_S with full precision (only if
          ``spec_edited == True``) and display the stored value in selected format
        """

        def _store_entry(source):
            if self.spec_edited:
                if source.objectName() == "stimFreq1":
                   self.f1 = safe_eval(source.text(), self.f1 * fb.fil[0]['f_S'],
                                            return_type='float') / fb.fil[0]['f_S']
                   source.setText(str(params['FMT'].format(self.f1 * fb.fil[0]['f_S'])))

                elif source.objectName() == "stimFreq2":
                   self.f2 = safe_eval(source.text(), self.f2 * fb.fil[0]['f_S'],
                                            return_type='float') / fb.fil[0]['f_S']
                   source.setText(str(params['FMT'].format(self.f2 * fb.fil[0]['f_S'])))

                self.spec_edited = False # reset flag
                self.sig_tx.emit({'sender':__name__, 'ui_changed':'stim'})

#        if isinstance(source, QLineEdit):
#        if source.objectName() in {"stimFreq1","stimFreq2"}:
        if event.type() in {QEvent.FocusIn,QEvent.KeyPress, QEvent.FocusOut}:
            if event.type() == QEvent.FocusIn:
                self.spec_edited = False
                self.load_fs()
            elif event.type() == QEvent.KeyPress:
                self.spec_edited = True # entry has been changed
                key = event.key()
                if key in {Qt.Key_Return, Qt.Key_Enter}:
                    _store_entry(source)
                elif key == Qt.Key_Escape: # revert changes
                    self.spec_edited = False
                    if source.objectName() == "stimFreq1":
                        source.setText(str(params['FMT'].format(self.f1 * fb.fil[0]['f_S'])))
                    elif source.objectName() == "stimFreq2":
                        source.setText(str(params['FMT'].format(self.f2 * fb.fil[0]['f_S'])))

            elif event.type() == QEvent.FocusOut:
                _store_entry(source)

        # Call base class method to continue normal event processing:
        return super(PlotImpzSpecgram_UI, self).eventFilter(source, event)

        
    # -------------------------------------------------------------------------

    def update_N(self, emit=True):
        # TODO: dict_sig not needed here, call directly from impz, distinguish
        # between local triggering and updates upstream
        """
        Update values for self.N and self.N_start from the QLineEditWidget,
        update the window and fire "data_changed"
        """
        if not isinstance(emit, bool):
            logger.error("update N: emit={0}".format(emit))
        self.N_start = safe_eval(self.led_N_start.text(), self.N_start, return_type='int', sign='poszero')
        self.led_N_start.setText(str(self.N_start)) # update widget
        self.N_user = safe_eval(self.led_N_points.text(), self.N_user, return_type='int', sign='poszero')

        if self.N_user == 0: # automatic calculation
            self.N = self.calc_n_points(self.N_user) # widget remains set to 0
            self.led_N_points.setText("0") # update widget
        else:
            self.N = self.N_user
            self.led_N_points.setText(str(self.N)) # update widget

        self.N_end = self.N + self.N_start # total number of points to be calculated: N + N_start

        # FFT window needs to be updated due to changed number of data points
        self._update_win_fft(emit=False) # don't emit anything here
        if emit:
            self.sig_tx.emit({'sender':__name__, 'ui_changed':'N'})


    def _read_param1(self):
        """Read out textbox when editing is finished and update dict and fft window"""
        param = safe_eval(self.ledWinPar1.text(), self.win_dict['par'][0]['val'],
                          return_type='float')
        if param < self.win_dict['par'][0]['min']:
            param = self.win_dict['par'][0]['min']
        elif param > self.win_dict['par'][0]['max']:
            param = self.win_dict['par'][0]['max']
        self.ledWinPar1.setText(str(param))
        self.win_dict['par'][0]['val'] = param
        self._update_win_fft()

    def _read_param2(self):
        """Read out textbox when editing is finished and update dict and fft window"""
        param = safe_eval(self.ledWinPar2.text(), self.win_dict['par'][1]['val'],
                          return_type='float')
        if param < self.win_dict['par'][1]['min']:
            param = self.win_dict['par'][1]['min']
        elif param > self.win_dict['par'][1]['max']:
            param = self.win_dict['par'][1]['max']
        self.ledWinPar2.setText(str(param))
        self.win_dict['par'][1]['val'] = param
        self._update_win_fft()


#------------------------------------------------------------------------------

def main():
    import sys
    from pyfda.libs.compat import QApplication

    app = QApplication(sys.argv)

    mainw = PlotImpzSpecgram_UI(None)
    layVMain = QVBoxLayout()
    layVMain.addWidget(mainw.wdg_ctrl_spec)

    layVMain.setContentsMargins(*params['wdg_margins'])#(left, top, right, bottom)

    mainw.setLayout(layVMain)


    app.setActiveWindow(mainw)
    mainw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

    # module test using python -m pyfda.plot_widgets.plot_impz_specgram_ui