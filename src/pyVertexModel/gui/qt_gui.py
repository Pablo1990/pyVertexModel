from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QCheckBox, QComboBox
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Set Functions")

        layout = QVBoxLayout()

        # Option to pick the type of model between the different models
        QCombobox = QComboBox()
        QCombobox.addItem("VertexModelVoronoiFromTimeImage")
        QCombobox.addItem("VertexModelVoronoiFromImage")
        QCombobox.addItem("Cyst")
        layout.addItem(QCombobox)

        # Option to pick the starting geometry

        vModel = VertexModelVoronoiFromTimeImage()
        vModel.initialize('/Users/pablovm/PostDoc/pyVertexModel/Tests/data/Newton_Raphson_Iteration_wingdisc.mat')
        # load_state(vModel, '/Users/pablovm/PostDoc/pyVertexModel/Result/04-10_132335_VertexModelTime_Cells_40_visc_500_lVol_0.001_muBulk_3000_lBulk_2000_kSubs_1_lt_0.05_noise_0.1_eTriAreaBarrier_0.5_eARBarrier_0.1_RemStiff_0.7_lS1_10_lS2_1.0_lS3_1.0/data_step_before_remodelling_7.pkl')
        # #vModel.set.RemodelStiffness = 0.6
        # vModel.iteration_converged()
        # # #vModel.t = 0
        # # #vModel.tr = 0
        # #vModel.reset_contractility_values()
        # vModel.iterate_over_time()

        self.set_instance = vModel.set

        widget = QWidget()

        self.NoBulk_110_button = QPushButton("NoBulk_110")
        self.NoBulk_110_button.clicked.connect(self.set_instance.NoBulk_110)
        layout.addWidget(self.NoBulk_110_button)

        self.stretch_button = QPushButton("stretch")
        self.stretch_button.clicked.connect(self.set_instance.stretch)
        layout.addWidget(self.stretch_button)

        self.cyst_button = QPushButton("cyst")
        self.cyst_button.clicked.connect(self.set_instance.cyst)
        layout.addWidget(self.cyst_button)

        widget.setLayout(layout)
        self.setCentralWidget(widget)