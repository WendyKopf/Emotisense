__author__ = 'Tomasz Ksepka'

from PyQt4 import QtGui, QtCore
import capture
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


class VideoFeedWindow(QtGui.QWidget):

    def pause(self):
        self.video.stop()

    def play(self):
        self.video.start()

    def __init__(self, parent=None):

        super(VideoFeedWindow, self).__init__(parent)
        
        self.graph_x = [0]
        self.frame_number = 0
        self.graph_y_neutral = [0]
        self.graph_y_angry_disgust = [0]
        self.graph_y_fear_surprise = [0]
        self.graph_y_happy = [0]
        self.graph_y_sad = [0]
        
        #self.graph_win = pg.GraphicsWindow(title="Basic plotting examples")
        #self.graph_win.resize(500,500)
        
        #self.plot = self.graph_win.addPlot(title="Emotions vs. Time",  y=np.random.normal(size=100))
        #self.plot.plot(np.random.normal(size=100), pen=(255,0,0), name="Neutral")
        
        #self.test_x = [0, 1, 2, 3, 4, 5, 6]
        #self.test_y = [1, 2, 3, 4, 5, 6, 7]
        #self.plot = plt.plot(self.text_x, self.test_y)
        #fig = plt.figure()
        #fig.add_subplot(self.plot)
        #fig.canvas.draw()
        #graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=uint8, sep='')
        #graph_np = graph_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        fig = plt.figure(figsize=(5,5))
        ax1 = fig.add_subplot(111)
        ax1.plot(self.graph_x, self.graph_y_neutral)
        fig.canvas.draw()
        #fig.show()
        
        #graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=uint8, sep='')
        #graph_np = graph_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        w,h = fig.canvas.get_width_height()
        graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_np.shape = (w, h, 3)
        
        #height, width = graph_np.shape[:2]
        image = QtGui.QImage(graph_np, w, h, QtGui.QImage.Format_RGB888)
        image = QtGui.QPixmap.fromImage(image)
        self.graph_img = image
        
        self.set_up()

    def set_up(self):
        self.setGeometry(50, 50, 800, 800)
        self.video = capture.Capture(cv2.VideoCapture(0), self)
        self.video.start()
        # self.video.show()

        font = QtGui.QFont('Times', 16, QtGui.QFont.Bold)

        video_feed_label = QtGui.QLabel('Video Feed', self)
        video_feed_label.setFont(font)
        video_feed_label.setAlignment(QtCore.Qt.AlignCenter)

        play_button = QtGui.QPushButton('Resume', self)
        play_button.clicked.connect(self.play)

        pause_button = QtGui.QPushButton('Pause', self)
        pause_button.clicked.connect(self.pause)

        main_layout = QtGui.QVBoxLayout()
        main_layout.setAlignment(QtCore.Qt.AlignTop)

        button_layout = QtGui.QHBoxLayout()
        button_layout.addWidget(play_button)
        button_layout.addWidget(pause_button)
        
        emotion_table = QtGui.QTableWidget()
        self.emotion_table = emotion_table
        emotion_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)

        emotion_table.setHorizontalHeaderLabels(QtCore.QString('Emotion;Count').split(';'))
        emotion_table.setRowCount(5)
        emotion_table.setColumnCount(2)
        
        emotion_table.setItem(0, 0, QtGui.QTableWidgetItem('Neutral'))
        emotion_table.setItem(1, 0, QtGui.QTableWidgetItem('Angry/Disgust'))
        emotion_table.setItem(2, 0, QtGui.QTableWidgetItem('Fear/Surprise'))
        emotion_table.setItem(3, 0, QtGui.QTableWidgetItem('Happy'))
        emotion_table.setItem(4, 0, QtGui.QTableWidgetItem('Sad'))
        
        # plot = pyqtgraph.PlotWidget()
        # plot.setXRange(0, 20, padding=.001)
        # plot.setYRange(0, 20, padding=.001)
        graph_label = QtGui.QLabel()
        self.graph_label = graph_label
        graph_label.setPixmap(self.graph_img)

        graph_layout = QtGui.QHBoxLayout()
        graph_layout.addWidget(emotion_table)
        
        
        graph_layout.addWidget(graph_label)
        
        # graph_layout.addWidget(plot)

        main_layout.addWidget(video_feed_label)
        main_layout.addWidget(self.video)
        main_layout.addLayout(graph_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        
    def update_graph(self,emotions):
        ## graph
        #print "updating graph"
        self.frame_number += 1
        self.graph_x.append(self.frame_number)
        self.graph_y_neutral.append(emotions[0])
        self.graph_y_angry_disgust.append(emotions[1])
        self.graph_y_fear_surprise.append(emotions[2])
        self.graph_y_happy.append(emotions[3])
        self.graph_y_sad.append(emotions[4])
        
        fig = plt.figure(figsize=(5,5))
        newplot = fig.add_subplot(111)
        newplot.plot(self.graph_x[-20:], self.graph_y_neutral[-20:], label = "Neutral")
        newplot.plot(self.graph_x[-20:], self.graph_y_angry_disgust[-20:], label = "Anger/Disgust")
        newplot.plot(self.graph_x[-20:], self.graph_y_fear_surprise[-20:], label = "Fear/Surprise")
        newplot.plot(self.graph_x[-20:], self.graph_y_happy[-20:], label = "Happy")
        newplot.plot(self.graph_x[-20:], self.graph_y_sad[-20:], label = "Sad")
        newplot.set_ylim((0,5))
        newplot.set_xlabel("Frame No")
        newplot.set_ylabel("Number of Emotions")
        newplot.legend()
        fig.canvas.draw()
        #fig.show()
        
        #graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=uint8, sep='')
        #graph_np = graph_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        w,h = fig.canvas.get_width_height()
        graph_np = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
        graph_np.shape = (w, h, 3)
        
        #height, width = graph_np.shape[:2]
        image = QtGui.QImage(graph_np, w, h, QtGui.QImage.Format_RGB888)
        image = QtGui.QPixmap.fromImage(image)
        self.graph_label.setPixmap(image)
        
        #print(self.graph_x)
        #print(self.graph_y_neutral)

    def update_table(self, emotions):
        print 'Emotion Count: ' + str(emotions)
        
        self.emotion_table.setItem(0, 1, QtGui.QTableWidgetItem(str(emotions[0])))
        self.emotion_table.setItem(1, 1, QtGui.QTableWidgetItem(str(emotions[1])))
        self.emotion_table.setItem(2, 1, QtGui.QTableWidgetItem(str(emotions[2])))
        self.emotion_table.setItem(3, 1, QtGui.QTableWidgetItem(str(emotions[3])))
        self.emotion_table.setItem(4, 1, QtGui.QTableWidgetItem(str(emotions[4])))
        