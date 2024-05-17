import ctypes
import sys
import cchardet as cchardet

from PyQt5.QtWidgets import *
from ui_designer.untitled import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, sip
from PyQt5.QtGui import QPixmap, QPainter, QImage, QPalette, QBrush, QFont, QColor
from PyQt5.QtCore import Qt, QDir

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")  #设置最小化工具栏图标

class MyMainWindow(QMainWindow, Ui_MainWindow): # 继承 QMainWindow 类和 Ui_MainWindow 界面类
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        #加载设置
        self.init_info()
        #显示上次打开的文件
        if self.current_file:
            self.load_file(self.current_file)

        self.actionfileopen.triggered.connect(self.open_file)



    ###init_info用于初始化文件设置，保存打开的文件###
    def init_info(self):
        self.setting = QtCore.QSettings("./config.ini", QtCore.QSettings.IniFormat)  # 配置文件
        self.setting.setIniCodec('utf-8')  # 设置配置文件的编码格式
        self.current_file = self.setting.value("FILE/file")  # 目前打开的文件
        self.history_files = self.setting.value("FILE/files")  # 最近打开的文件
        if not self.history_files:
            self.history_files = []

    ###save_info函数用于保存当前打开的文件和记录历史文件
    def save_info(self):
        self.setting.setValue("FILE/file", self.current_file)
        self.setting.setValue("FILE/files", self.history_files)

    ###open_file方法用于查找和打开文件
    def open_file(self):
        #弹出选择文件的窗口，getOpenFileName()方法返回的是一个元组，其中第一个元素是用户选择的文件路径，第二个元素是文件选择对话框的状态。
        if not self.current_file:
            path = QDir.homePath()
        else:
            path = self.current_file

        fname = QFileDialog.getOpenFileName(self, '打开文件', path, 'Text Files (*.txt)')
        self.load_file(fname[0])

    def load_file(self, file):
        #try 块用于捕获和处理在程序执行过程中可能发生的异常。使用 except 块来处理可能发生的异常。
        if file:
            try:
                #更改目前打开的文件
                self.current_file = file
                self.filename = file.split('/')[-1].split('.')[0]  #先将路径按'/'分割并取最后一个元素通常为文件名即'xxx.txt'，再按点分割，取第一个

                #使用最久未使用算法，更新最近打开的文件'history_file'
                if file in self.history_files:
                    self.history_files.remove(file)  #先移除队列，在添加元组末尾
                self.history_files.append(file)

                #书架中存储最近打开的十本书
                if len(self.history_files) > 10:
                    self.history_files.pop(0)  #弹出最久未使用的书

                #获取文件的编码格式（由于不知道文件的编码）
                encoding_format = self.get_encoding_format(file)
                with open(file, 'r', encoding = encoding_format) as f:  #使用上下文管理器打开文件，file为文件路径，‘r'表示读取模式（只读），encoding指定编码格式
                    txt = f.read()
                    self.textBrowser.setText(txt)
                    self.textBrowser.setStatusTip(self.filename)
                    self.display_files() #显示最近的文件

            except:
                self.show_msg('文件不存在或读取时发生未知错误！')

        else:
            self.show_msg('您没有选择文件或取消了操作！')

    #获取文本文件编码
    def get_encoding_format(self,file):
        with open(file, 'rb') as f:
            return cchardet.detect(f.read())['encoding']

    def show_msg(self, msg):
        # 后两项分别为按钮(以|隔开，共有7种按钮类型，见示例后)、默认按钮(省略则默认为第一个按钮)
        reply = QMessageBox.information(self, "提示", msg, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

    #重写closeevent函数
    def closeEvent(self, event):
        result = QMessageBox.information(self, "退出应用", "确认退出应用？", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            #关闭窗口时保存应用设置
            self.save_info()    #关闭窗口时保存应用设置
            event.accept()      #接收事件允许窗口的关闭
            QMainWindow.closeEvent(self, event)
        else:
            event.ignore()      #忽略时间阻止窗口的关闭

    def display_files(self):
        #显示最近打开的文件
        #每一次打开重新显示（根据最近使用算法排列）
        _translate = QtCore.QCoreApplication.translate
        self.filesmenu.clear()
        for i, file in enumerate(self.history_files):
            name = file.split('/')[-1].split('.')[0]     #获取文本文件名
            action = QtWidgets.QAction(self)
            action.setObjectName(f'file{i}')
            
            self.filesmenu.addAction(action)      #给历史文件添加动作
            action.setText(_translate("MyMainWindow", name))
            action.triggered.connect(self.open_history_files)

    def open_history_files(self):
        sender = self.sender().objectName()
        self.load_file(self.history_files[int(sender[-1])])




if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序
