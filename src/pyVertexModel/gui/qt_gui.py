from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QCheckBox, QComboBox, QLabel, \
    QStackedLayout, QFileDialog, QFormLayout, QLineEdit, QGroupBox

from src.pyVertexModel.algorithm.vertexModelBubbles import VertexModelBubbles
from src.pyVertexModel.algorithm.vertexModelVoronoiFromTimeImage import VertexModelVoronoiFromTimeImage
from src.pyVertexModel.util.utils import load_state


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.advanced_layout = None
        self.advanced_groupbox = None
        self.basic_layout = None
        self.vModel = VertexModelVoronoiFromTimeImage()
        self.run_button = None
        self.new_simulation_widget = None
        self.select_file_button = None
        self.load_state_widget = None
        self.selected_filename = None
        self.setWindowTitle("Simulation Options")

        self.layout = QVBoxLayout()

        self.switch = QComboBox()
        self.switch.addItem("New simulation")
        self.switch.addItem("Load state")
        self.switch.currentIndexChanged.connect(self.update_layout)
        self.layout.addWidget(self.switch)

        # Add widgets for new simulation layout
        self.add_widgets_new_simulation()

        # Add a "Run Simulation" button
        self.add_run_button()

        self.widget = QWidget()
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def add_widgets_new_simulation(self):
        self.new_simulation_widget = QLabel("New simulation layout")
        self.layout.addWidget(self.new_simulation_widget)
        self.add_type_of_model_widget()

        # Basic attributes
        self.basic_layout = QFormLayout()
        self.basic_layout.addRow("TotalCells", QLineEdit(str(self.vModel.set.TotalCells)))
        self.basic_layout.addRow("CellHeight", QLineEdit(str(self.vModel.set.CellHeight)))

        self.basic_layout.addRow("lambdaV", QLineEdit(str(self.vModel.set.lambdaV)))
        self.basic_layout.addRow("kSubstrate", QLineEdit(str(self.vModel.set.kSubstrate)))
        self.basic_layout.addRow("cLineTension", QLineEdit(str(self.vModel.set.cLineTension)))
        self.basic_layout.addRow("noiseContractility", QLineEdit(str(self.vModel.set.noiseContractility)))
        self.basic_layout.addRow("lambdaS1", QLineEdit(str(self.vModel.set.lambdaS1)))

        self.layout.addLayout(self.basic_layout)

        # Advanced attributes
        self.advanced_groupbox = QGroupBox("Advanced")
        self.advanced_groupbox.setCheckable(True)
        self.advanced_groupbox.setChecked(False)

        self.advanced_layout = QFormLayout()
        self.advanced_layout.addRow("lambdaS2", QLineEdit(str(self.vModel.set.lambdaS2)))
        self.advanced_layout.addRow("lambdaS3", QLineEdit(str(self.vModel.set.lambdaS3)))
        # Add more advanced attributes here...

        self.advanced_groupbox.setLayout(self.advanced_layout)

        self.layout.addWidget(self.advanced_groupbox)

    def add_type_of_model_widget(self):
        QCombobox = QComboBox()
        QCombobox.addItem("Voronoi from time images")
        QCombobox.addItem("Cyst")
        QCombobox.currentIndexChanged.connect(self.initialize_different_model)
        QCombobox.setCurrentIndex(0)
        self.layout.addWidget(QCombobox)

    def add_widgets_load_state_layout(self):
        self.load_state_widget = QLabel("Load state layout")
        self.layout.addWidget(self.load_state_widget)
        self.add_type_of_model_widget()
        self.select_file_button = QPushButton("Select File")
        self.select_file_button.clicked.connect(self.select_file)
        self.layout.addWidget(self.select_file_button)

    def add_run_button(self):
        """
        Add a "Run Simulation" button to the layout
        :return: 
        """
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation)
        self.layout.addWidget(self.run_button)

    def update_layout(self, index):
        """
        Update the layout based on the selected option
        :param index:
        :return:
        """
        while self.layout.count():
            child = self.layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

            # Add the switch back to the layout
        self.layout.addWidget(self.switch)

        # Add widgets to the layout based on the selected option
        if index == 0:  # New simulation
            # Add widgets for new simulation
            self.add_widgets_new_simulation()
        elif index == 1:  # Load state
            # Add widgets for load state
            self.add_widgets_load_state_layout()

        self.add_run_button()

    def load_state(self):
        """
        Load state of the model
        :return:
        """
        load_state(self.vModel, self.selected_filename)

    def select_file(self):
        """
        Open a file dialog to select a file
        :return:
        """
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.selected_filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                                "All Files (*);;Pkl Files (*.pkl);;Mat files (*.mat)",
                                                                options=options)
        if self.selected_filename:
            load_state(self.vModel, self.selected_filename)
            self.selected_filename = None

    def initialize_different_model(self, index):
        """
        Initialize different models based on the index
        :param index:
        :return:
        """
        if index == 0:
            self.vModel = VertexModelVoronoiFromTimeImage()
        elif index == 1:
            self.vModel = VertexModelBubbles()

    def run_simulation(self):
        """
        Run the simulation
        :return:
        """
        self.vModel.iterate_over_time()

        # # Option to pick the type of model between the different models
        #
        # # Option to pick the starting geometry
        #
        # vModel = VertexModelVoronoiFromTimeImage()
        # vModel.initialize('/Users/pablovm/PostDoc/pyVertexModel/Tests/data/Newton_Raphson_Iteration_wingdisc.mat')
        # #
        # # #vModel.set.RemodelStiffness = 0.6
        # # vModel.iteration_converged()
        # # # #vModel.t = 0
        # # # #vModel.tr = 0
        # # #vModel.reset_contractility_values()
        # # vModel.iterate_over_time()
