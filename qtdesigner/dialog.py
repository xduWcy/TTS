# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(960, 142)
        Dialog.setMinimumSize(QtCore.QSize(960, 142))
        Dialog.setMaximumSize(QtCore.QSize(960, 142))
        Dialog.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icon/audiobook2.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Dialog.setWindowIcon(icon)
        Dialog.setToolTip("")
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalSlider = QtWidgets.QSlider(Dialog)
        self.verticalSlider.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.verticalSlider.setMaximum(100)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.verticalLayout.addWidget(self.verticalSlider, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.pushButton_5 = QtWidgets.QPushButton(Dialog)
        self.pushButton_5.setMinimumSize(QtCore.QSize(30, 30))
        self.pushButton_5.setMaximumSize(QtCore.QSize(30, 30))
        self.pushButton_5.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_5.setWhatsThis("")
        self.pushButton_5.setStyleSheet("QPushButton{\n"
"background-color:rgba(255,255,255,0);      \n"
"color:rgb(0,0,0);                           \n"
"border-radius:15px;                          \n"
"border: 3px outset rgba(255,255,255,0);         \n"
"font:bold 10px;                              \n"
"}\n"
" \n"
"\n"
"QPushButton:hover{\n"
"background-color:rgba(150,150,150,150);\n"
"color:rgb(0,255，255);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"background-color:rgba(150,150,150,200);\n"
"color:rgb(0,255，255);\n"
"}\n"
" \n"
" \n"
" ")
        self.pushButton_5.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icon/volume-notice.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_5.setIcon(icon1)
        self.pushButton_5.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalSlider_2 = QtWidgets.QSlider(Dialog)
        self.verticalSlider_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.verticalSlider_2.setMaximum(300)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.verticalLayout_2.addWidget(self.verticalSlider_2, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setMinimumSize(QtCore.QSize(30, 30))
        self.pushButton.setMaximumSize(QtCore.QSize(30, 30))
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("QPushButton{\n"
"background-color:rgba(255,255,255,0);      \n"
"color:rgb(0,0,0);                           \n"
"border-radius:15px;                          \n"
"border: 3px outset rgba(255,255,255,0);         \n"
"font:bold 10px;                              \n"
"}\n"
" \n"
"\n"
"QPushButton:hover{\n"
"background-color:rgba(150,150,150,150);\n"
"color:rgb(0,255，255);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"background-color:rgba(150,150,150,200);\n"
"color:rgb(0,255，255);\n"
"}\n"
" \n"
" \n"
" ")
        self.pushButton.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icon/speed (2).svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon2)
        self.pushButton.setIconSize(QtCore.QSize(30, 30))
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout_2.addWidget(self.pushButton, 0, QtCore.Qt.AlignHCenter)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setMinimumSize(QtCore.QSize(30, 30))
        self.pushButton_2.setMaximumSize(QtCore.QSize(30, 30))
        self.pushButton_2.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_2.setStyleSheet("QPushButton{\n"
"background-color:rgba(255,255,255,0);      \n"
"color:rgb(0,0,0);                           \n"
"border-radius:15px;                          \n"
"border: 3px outset rgba(255,255,255,0);         \n"
"font:bold 10px;                              \n"
"}\n"
" \n"
"\n"
"QPushButton:hover{\n"
"background-color:rgba(150,150,150,150);\n"
"color:rgb(0,255，255);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"background-color:rgba(150,150,150,200);\n"
"color:rgb(0,255，255);\n"
"}\n"
" \n"
" \n"
" ")
        self.pushButton_2.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icon/setting-two.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon3)
        self.pushButton_2.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2, 0, QtCore.Qt.AlignBottom)
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setMinimumSize(QtCore.QSize(0, 30))
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 11111111))
        self.comboBox.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.comboBox.setStyleSheet("QComboBox{\n"
"  color:#666666;\n"
"  font-size:14px;\n"
"  padding: 1px 15px 1px 3px;\n"
"  border:1px solid rgba(228,228,228,1);\n"
"  border-radius:5px 5px 0px 0px;\n"
"} \n"
" 下拉按钮QComboBox::drop-down ，可以设置按钮的位置，大小、背景图，边框等\n"
" \n"
"  QComboBox::drop-down {\n"
"      subcontrol-origin: padding;\n"
"      subcontrol-position: top right;\n"
"      width: 15px;\n"
"      border:none;\n"
"  }\n"
"箭头图标 QComboBox::down-arrow这个很简单就是把自己箭头加进去就行。\n"
" \n"
"  QComboBox::down-arrow {\n"
"      image: url(:/res/work/dateDown.png);\n"
"  }\n"
"下拉列表QComboBox QAbstractItemView，因为QComboBox的view是QAbstractItemView的子类，所以是后代迭代器的写法。\n"
"  \n"
" QComboBox QAbstractItemView{\n"
"    background:rgba(255,255,255,1);\n"
"    border:1px solid rgba(228,228,228,1);\n"
"    border-radius:0px 0px 5px 5px;\n"
"    font-size:14px;\n"
"    outline: 0px;  //去虚线\n"
"  }")
        self.comboBox.setIconSize(QtCore.QSize(10, 10))
        self.comboBox.setObjectName("comboBox")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icon/waves-left.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.comboBox.addItem(icon4, "")
        self.comboBox.addItem("")
        self.horizontalLayout.addWidget(self.comboBox, 0, QtCore.Qt.AlignBottom)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 2, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(Dialog)
        self.pushButton_3.setMinimumSize(QtCore.QSize(30, 30))
        self.pushButton_3.setMaximumSize(QtCore.QSize(30, 30))
        self.pushButton_3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton_3.setStyleSheet("QPushButton{\n"
"background-color:rgba(255,255,255,0);      \n"
"color:rgb(0,0,0);                           \n"
"border-radius:15px;                          \n"
"border: 3px outset rgba(255,255,255,0);         \n"
"font:bold 10px;                              \n"
"}\n"
" \n"
"\n"
"QPushButton:hover{\n"
"background-color:rgba(150,150,150,150);\n"
"color:rgb(0,255，255);\n"
"}\n"
"\n"
"QPushButton:pressed{\n"
"background-color:rgba(150,150,150,200);\n"
"color:rgb(0,255，255);\n"
"}\n"
" \n"
" \n"
" ")
        self.pushButton_3.setText("")
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icon/play (1).svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon5)
        self.pushButton_3.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout.addWidget(self.pushButton_3, 0, 3, 1, 1, QtCore.Qt.AlignBottom)
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 0, 4, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "懒人听书"))
        self.pushButton_5.setToolTip(_translate("Dialog", "volume"))
        self.pushButton.setToolTip(_translate("Dialog", "speed"))
        self.pushButton_2.setToolTip(_translate("Dialog", "Dubbing Type"))
        self.comboBox.setItemText(0, _translate("Dialog", "11"))
        self.comboBox.setItemText(1, _translate("Dialog", "22"))
        self.pushButton_3.setToolTip(_translate("Dialog", "play"))
import qtdesigner.resource_rc