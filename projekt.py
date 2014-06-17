from pylab import * 
import sys
import NN as nn
import math
from PySide.QtCore import *
from PySide.QtGui import *

class MainWindow(QDialog):
    def __init__(self):
        self.train_data = [] #initialization of training data

        super(MainWindow, self).__init__()

        # initialization of elements on the GUI:
        ## Training part
        iNodesLabel = QLabel('Input nodes: ')
        oNodesLabel = QLabel('Output nodes: ')
        self.iNodesSB = QSpinBox()
        self.oNodesSB = QSpinBox()
        generateNN = QPushButton('Generate NN')
        self.NNLabel = QLabel()
        self.TrainDataLabel = QLabel()
        selectTrainData = QPushButton('Load Training Data')
        trainNN = QPushButton('Train')
        self.numIter = QSpinBox()
        self.numIter.setRange(1000, 10000)
        nuIterLabel = QLabel('Number of iterations: ')
        self.trainError = QLabel()
        finalErrorRateLabel = QLabel('Final Error Rate:')
        self.finalErrorRate = QLabel()
        saveNN = QPushButton("Save NN")

        ## Testing part
        loadNNButton = QPushButton("Load NN")
        self.loadNNLabel = QLabel()
        loadTestDataButton = QPushButton("Load Test Data")
        self.loadTestData = QLabel()
        testButton = QPushButton("Test!")
        self.testBrowser = QTextBrowser()



        self.setWindowTitle('Simple ANN')
        

        #### LAYOUT ####
        # Training part
        layout = QGridLayout()
        layout.addWidget(iNodesLabel, 0, 0)
        layout.addWidget(self.iNodesSB, 0, 1)
        layout.addWidget(oNodesLabel, 1, 0)
        layout.addWidget(self.oNodesSB, 1, 1)

        layout.addWidget(generateNN, 2, 0, 1, 2)

        layout.addWidget(self.NNLabel, 3, 0, 1, 2)

        layout.addWidget(selectTrainData, 4, 0)
        layout.addWidget(self.TrainDataLabel, 4, 1)

        layout.addWidget(nuIterLabel, 5, 0)
        layout.addWidget(self.numIter, 5, 1)
        
        layout.addWidget(trainNN, 6, 0)
        layout.addWidget(self.trainError, 6, 1)

        layout.addWidget(finalErrorRateLabel, 7, 0)
        layout.addWidget(self.finalErrorRate, 7, 1)

        layout.addWidget(saveNN)

        # Testing part
        layout.addWidget(loadNNButton, 0, 4)
        layout.addWidget(self.loadNNLabel, 0, 5)

        layout.addWidget(loadTestDataButton, 1, 4)
        layout.addWidget(self.loadTestData, 1, 5)

        layout.addWidget(testButton, 2, 4)

        layout.addWidget(self.testBrowser, 3, 4, 5, 3)



        self.setLayout(layout)


        #### SIGNALS ####
        self.connect(generateNN, SIGNAL("clicked()"), self.generate_NN)
        self.connect(selectTrainData, SIGNAL("clicked()"), self.load_train_data)
        self.connect(trainNN, SIGNAL("clicked()"), self.train_nn)
        self.connect(saveNN, SIGNAL("clicked()"), self.save_NN)

        self.connect(loadNNButton, SIGNAL("clicked()"), self.load_NN)
        self.connect(loadTestDataButton, SIGNAL("clicked()"), self.load_test_data)
        self.connect(testButton, SIGNAL("clicked()"), self.test)

    def generate_NN(self):
        inputNodes = self.iNodesSB.value()
        outputNodes = self.oNodesSB.value()
        hiddenNodes = int(math.ceil((inputNodes + outputNodes)/2.0))

        self.NNValues = [inputNodes, hiddenNodes, outputNodes] # needed for saving NN later on

        self.nn = nn.NN(inputNodes, hiddenNodes, outputNodes)
        self.NNLabel.setText(("<font color='green'><i>Generated NN (%d,%d,%d)</i></font>") % (inputNodes, hiddenNodes, outputNodes))

    def load_train_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File')
        f = open(fname, 'r')
        try:
            for line in f:
                SB , UZ, TAR = line.split('\t')
                self.train_data.append([self.normaliziraj(int(SB), int(UZ)), [int(TAR)]])
            self.TrainDataLabel.setText('<font color="green"><i>Data loaded!</i></font>')
        except:
            self.TrainDataLabel.setText('<font color="red"><i>Wrong format!</i></font>')


    def normaliziraj(self, sb, uz):
        return [float(sb)/5, float(uz)/100]

    def train_nn(self):
        numIterValue = self.numIter.value()
        try:
            error_rate = self.nn.train(self.train_data, numIterValue)
            self.finalErrorRate.setText(str("%.5f" % error_rate[-1]))
            self.plot_error_rate(error_rate)
        except:
            self.trainError.setText('<font color="red"><i>Wrong values!</i></font>')

    
    def plot_error_rate(self, error):
        X = range(0,len(error))
        Y = error
        plot( X, Y)
        xlabel('Time')
        ylabel('Error')
        title('Error rate')
        grid(True)
        show()

    def save_NN(self):
        format = "*.nn"
        filename = QFileDialog.getSaveFileName(self, "Save NN as", '.', format)[0]
        fn = "%s.nn" % filename
        self.nn.save_nn(fn, self.NNValues)

    def load_NN(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File', ".", "*.nn")
        f = open(fname, 'r')
        br_input = int(f.readline().strip('\n'))
        br_hidden = int(f.readline().strip('\n'))
        br_output = int(f.readline().strip('\n'))

        inputWeights = []
        for i in range(br_input + 1):
            temp = f.readline().strip('\n')
            temp = temp.split(',')
            temp = [float(i) for i in temp]
            inputWeights.append(temp)

        outputWeights = []
        for i in range(br_hidden):
            temp = f.readline().strip('\n')
            outputWeights.append(float(temp))
        f.close()

        self.net = nn.NN(br_input, br_hidden, br_output)
        self.net.set_weights(inputWeights, outputWeights)
        self.loadNNLabel.setText("<font color='green'><i>NN loaded!</i></font>")

    def load_test_data(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Open File')
        self.test_set = []

        f = open(fname, 'r')
        for line in f:
            SB , UZ, TAR = line.split('\t')
            self.test_set.append([self.normaliziraj(int(SB), int(UZ)), [int(TAR)]]) 
        self.loadTestData.setText("<font color='green'><i>Test data loaded!</i></font>")

    def test(self):
        results = self.net.test(self.test_set)

        for el in results:
            self.testBrowser.append(str(el))

app = QApplication(sys.argv)
form = MainWindow()
form.show()
app.exec_()