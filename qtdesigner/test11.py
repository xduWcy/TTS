import ctypes
import re
import sys
import cchardet as cchardet
import pyttsx3
import time
import pyttsx3
import  threading
import _thread as th
from PyQt5.QtWidgets import *
from untitled import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets, sip
from PyQt5.QtGui import QPixmap, QPainter, QImage, QPalette, QBrush, QFont, QColor
from PyQt5.QtCore import Qt, QDir
from dialog import Ui_Dialog
from TTS.tts_engine import *

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("myappid")  #设置最小化工具栏图标

class MyMainWindow(QMainWindow, Ui_MainWindow): # 继承 QMainWindow 类和 Ui_MainWindow 界面类
    def __init__(self, parent = None):
        super(MyMainWindow, self).__init__(parent)  # 初始化父类
        self.setupUi(self)  # 继承 Ui_MainWindow 界面类
        #加载设置
        self.init_info()
        self.pushButton.setHidden(True)
        self.pushButton_2.setHidden(True)
        # 设置默认背景色
        self.treeWidget.setStyleSheet(
            f"background-color: rgba({self.bg_color[0]}, {self.bg_color[1]}, {self.bg_color[2]},0.5);color: rgba({self.font_color[0]}, {self.font_color[1]}, {self.font_color[2]},0.5);border: 1px solid #000000;border-color: rgb({self.bg_color[0]}, {self.bg_color[1]}, {self.bg_color[2]})")
        self.textBrowser.setStyleSheet(
            f"background-color: rgba({self.bg_color[0]}, {self.bg_color[1]}, {self.bg_color[2]},0.5);color: rgba({self.font_color[0]}, {self.font_color[1]}, {self.font_color[2]},0.5);border: 1px solid #000000;border-color: rgb({self.bg_color[0]}, {self.bg_color[1]}, {self.bg_color[2]})")

        self.setAutoFillBackground(True)
        # 设置目录栏初始为隐藏状态
        self.treeWidget.setHidden(False)

        self.isPlaying = False

        #显示上次打开的文件
        if self.current_file:
            self.load_file(self.current_file)

        self.actionfileopen.triggered.connect(self.open_file)
        self.pushButton.clicked.connect(self.show_last)
        self.pushButton_2.clicked.connect(self.show_next)
        self.actionchoosetype.triggered.connect(self.select_font)
        self.actionbgcolor.triggered.connect(self.select_bgcolor)
        self.actioninsertbgpic.triggered.connect(self.select_pic)
        self.actioncolosebgpicture.triggered.connect(self.close_bg)
        self.actionreset.triggered.connect(self.default)
        self.actionhidemenu.triggered.connect(self.hide_catlog)
        self.actionread_books.triggered.connect(self.play)


    ###init_info用于初始化文件设置，保存打开的文件###
    def init_info(self):
        self.setting = QtCore.QSettings("./config.ini", QtCore.QSettings.IniFormat)  # 配置文件
        self.setting.setIniCodec('utf-8')  # 设置配置文件的编码格式
        self.current_file = self.setting.value("FILE/file")  # 目前打开的文件
        self.history_files = self.setting.value("FILE/files")  # 最近打开的文件
        if not self.history_files:
            self.history_files = []
        self.chapter = int(self.setting.value("FILE/chapter"))  # 上次浏览的章节
        self.fonts = self.setting.value("FONT/font")    # 字体
        self.fontsize = int(self.setting.value("FONT/fontsize")) # 字体大小
        self.bg_color = self.setting.value("BACKGROUND/color")  # 背景颜色
        self.bg = self.setting.value("BACKGROUND/bg")  # 背景图片
        self.windowSize = self.setting.value("SCREEN/screen")  # 窗口大小
        self.font_color = self.setting.value("FONT/fontcolor")  # 背景颜色

    ###save_info函数用于保存当前打开的文件和记录历史文件
    def save_info(self):
        self.setting.setValue("FILE/file", self.current_file)
        self.setting.setValue("FILE/files", self.history_files)
        self.setting.setValue("FILE/chapter", self.chapter)
        self.setting.setValue("FONT/font", self.fonts)
        self.setting.setValue("FONT/fontsize", self.fontsize)
        self.setting.setValue("BACKGROUND/color", self.bg_color)
        self.setting.setValue("BACKGROUND/bg", self.bg)
        self.setting.setValue("SCREEN/screen", self.windowSize)
        self.setting.setValue("FONT/fontcolor", self.font_color)

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
                if not self.history_files or self.current_file != self.history_files[-1]:
                    self.chapter = 0
                self.current_file = file
                self.filename = file.split('/')[-1].split('.')[0]  #先将路径按'/'分割并取最后一个元素通常为文件名即'xxx.txt'，再按点分割，取第一个

                #使用最久未使用算法，更新最近打开的文件'history_file'
                if file in self.history_files:
                    self.history_files.remove(file)  #先移除队列，在添加元组末尾
                self.history_files.append(file)

                #书架中存储最近打开的十本书
                if len(self.history_files) > 3:
                    self.history_files.pop(0)  #弹出最久未使用的书

                #获取文件的编码格式（由于不知道文件的编码）
                encoding_format = self.get_encoding_format(file)

                with open(file, 'r', encoding=encoding_format) as f:  #使用上下文管理器打开文件，file为文件路径，‘r'表示读取模式（只读），encoding指定编码格式
                    self.chapters = []                       #打开文件，生成章节目录
                    self.lines = f.readlines()               #按行读入数组，每一行算一个元组
                    chapterMatch = r"(第)([\u4e00-\u9fa5a-zA-Z0-9]{1,7})[章|节]"     #章节标题格式：第xxxx章\节
                    for i in range(len(self.lines)):
                        line = self.lines[i].strip()
                        if line != "" and re.match(chapterMatch, line):
                            line = line.replace("\n", "").replace("=", "")
                            if len(line) < 30:
                                self.chapters.append({line: i})
                #如果没有可用目录，就显示全部
                if not self.chapters:
                    self.chapters.append({self.filename:0})
                self.display_history_files() #显示最近的文件
                self.setup_chapters()    #设置章节目录
                self.show_content()      #设置文本显示器txt—browser的内容
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
            sys.exit(0)
        else:
            event.ignore()      #忽略时间阻止窗口的关闭

    def display_history_files(self):
        #显示最近打开的文件
        #每一次打开重新显示（根据最近使用算法排列）
        self.filesmenu.clear()
        _translate = QtCore.QCoreApplication.translate
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

    def setup_chapters(self):
        self.treeWidget.clear()
        _translate = QtCore.QCoreApplication.translate
        __sortingEnable = self.treeWidget.isSortingEnabled()
        for i, value in enumerate(self.chapters):
            item = QTreeWidgetItem(self.treeWidget)
            item.setText(0, _translate("MyMainWindow", list(value.keys())[0]))
            self.treeWidget.addTopLevelItem(item)                     #将新创建的树节点作为顶级项
        self.treeWidget.setSortingEnabled(__sortingEnable)
        self.treeWidget.clicked.connect(self.onTreeClicked)
        self.treeWidget.setCurrentItem(self.treeWidget.topLevelItem(self.chapter), 0)
        # 为当前章节设置背景色
        #self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(15, 136, 235))

    # 点击目录跳转到章节
    def onTreeClicked(self, index):
        # 恢复原来章节的背景色(设置透明度为0)，为新章节设置背景色

        top_level_item = self.treeWidget.topLevelItem(self.chapter)
        if top_level_item is not None:
            top_level_item.setBackground(0, QColor(0, 0, 0, 0))
        else:
            self.chapter = 0
            top_level_item = self.treeWidget.topLevelItem(self.chapter)
            top_level_item.setBackground(0, QColor(0, 0, 0, 0))
            print(f"No top-level item at index {self.chapter}")
        self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(0, 0, 0, 0))
        # 获取点击的项目下标
        self.chapter = int(index.row())
        # 判断按钮是否要显示
        self.button()
        self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(15, 136, 235))
        self.show_content()


    def show_content(self):
        self.button()
        self.textBrowser.setText(self.get_content())            #将文本内容加入到文本浏览器
        #self.new_window.refresh(self.lines, self.chapter, self.sta, self.en)
        self.textBrowser.setFont(QFont(self.fonts, self.fontsize))

        #self.textBrowser.setStatusTip(self.filename + "   " + list(self.chapters[self.chapter].keys())[0])    # 状态栏显示当前的章节内容和目录名

        # 获取章节内容
    def get_content(self):
        #index = self.chapter
        # 起始行
        if self.chapter > len(self.chapters) and self.chapter != 0:
            index = 0
        else:
            index = self.chapter


        start = list(self.chapters[index].values())[0]
        self.sta = start
        # 如果是终章
        if index == self.treeWidget.topLevelItemCount() - 1:
            return "".join(self.lines[start:-1])
        else:
            # 终止行
            end = list(self.chapters[index + 1].values())[0]
            self.en = end
            return "".join(self.lines[start:end])

    def show_last(self):
        self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(0, 0, 0, 0))
        self.chapter = self.chapter - 1
        self.show_content()  # 显示内容
        self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(15, 136, 235))

    def show_next(self):
        self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(0, 0, 0, 0))
        self.chapter = self.chapter + 1
        self.show_content()  # 显示内容
        self.treeWidget.topLevelItem(self.chapter).setBackground(0, QColor(15, 136, 235))

    def button(self):
        if len(self.chapters) == 1:
            self.pushButton.setHidden(True)
            self.pushButton_2.setHidden(True)
            # 第一章
        elif self.chapter == 0:
            self.pushButton.setHidden(True)
            self.pushButton_2.setVisible(True)
            # 末章
        elif self.chapter == len(self.chapters) - 1:
            self.pushButton.setVisible(True)
            self.pushButton_2.setHidden(True)
            # 其他情况，恢复按钮
        else:
            if self.pushButton.isHidden():
                self.pushButton.setVisible(True)
            if self.pushButton_2.isHidden():
                self.pushButton_2.setVisible(True)

    def select_font(self):
        # 弹出一个字体选择对话框。getFont()方法返回一个字体名称和状态信息。
        # 状态信息有OK和其他两种。
        font, ok = QFontDialog.getFont(QFont(self.fonts, self.fontsize), self, '选择字体和大小')
        # 如果点击OK，标签的字体就会随之更改
        if ok:
            self.textBrowser.setFont(font)
            self.fonts = font.family()
            self.fontsize = font.pointSize()

    # 选择背景颜色
    def select_bgcolor(self):
        #选择字体颜色
        font_col = QColorDialog.getColor(self.textBrowser.textColor(), self, "设置字体颜色")
        if font_col.isValid():
            print(font_col)
            #选择背景颜色
            bg_color = QColorDialog.getColor(Qt.white, self, "设置背景颜色")
            if bg_color.isValid():
                # 设置QTextBrowser的样式表来改变文本和背景颜色
                self.treeWidget.setStyleSheet(
                    f"background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()},0.5);"
                    f"border: 1px solid #000000;border-color: rgb({bg_color.red()}, {bg_color.green()}, {bg_color.blue()});"
                    f"color: rgba({font_col.red()}, {font_col.green()}, {font_col.blue()},0.5)")
                self.textBrowser.setStyleSheet(
                    f"background-color: rgba({bg_color.red()}, {bg_color.green()}, {bg_color.blue()},0.5);"
                    f"border: 1px solid #000000;border-color: rgb({bg_color.red()}, {bg_color.green()}, {bg_color.blue()});"
                    f"color: rgba({font_col.red()}, {font_col.green()}, {font_col.blue()},0.5)"
                )
                self.bg_color = [bg_color.red(), bg_color.green(), bg_color.blue()]
                self.font_color = [font_col.red(), font_col.green(), font_col.blue()]


    #背景图片
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), QPixmap(self.bg))
        super().paintEvent(event)

    # 选择背景图片
    def select_pic(self):
        if not self.bg:
            path = '/'
        else:
            path = self.bg
        fname = QFileDialog.getOpenFileName(self, '选择图片', path, filter='*.jpg,*.png,*.jpeg,All Files(*)')
        # 文件不为空
        if fname[0]:
            try:
                # 更改目前的背景图片
                self.bg = fname[0]
                self.update()   # 刷新页面
            except:
                self.show_msg('文件不存在或者错误！')
        else:  # 文件为空，说明没有选择文件
            self.show_msg('您没有选择文件！')

    # 关闭背景图片
    def close_bg(self):
        self.bg = ''    # 将背景图片设置为空即可
        self.update()   # 刷新页面

    # 窗口移动事件，保存用户最后设置的窗口大小
    def resizeEvent(self, event):
        self.windowSize = [event.size().width(), event.size().height()]

    # 恢复默认设置
    def default(self):
        result = QMessageBox.question(self, "恢复默认设置", "确定恢复默认设置？",
                                      QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            # 重新配置设置文件
            self.setting.setValue("SCREEN/screen", self.setting.value("DEFAULT/screen"))
            self.setting.setValue("FILE/file", self.setting.value("DEFAULT/file"))
            self.setting.setValue("FILE/files", self.setting.value("DEFAULT/files"))
            self.setting.setValue("BACKGROUND/bg", self.setting.value("DEFAULT/bg"))
            self.setting.setValue("BACKGROUND/color", self.setting.value("DEFAULT/color"))
            self.setting.setValue("FONT/font", self.setting.value("DEFAULT/font"))
            self.setting.setValue("FONT/fontsize", self.setting.value("DEFAULT/fontsize"))
            self.setting.setValue("FONT/fontcolor", self.setting.value("DEFAULT/fontcolor"))
            QtCore.QCoreApplication.instance().quit()

    # 显示/隐藏目录
    def hide_catlog(self):
        if self.treeWidget.isVisible():
            self.treeWidget.setHidden(True)
        else:
            self.treeWidget.setVisible(True)

    def play(self):
        #from mediapalyer_test import AnotherWindow
        self.new_window = AnotherWindow()
        self.new_window.refresh(self.lines, self.sta, self.en)
        self.new_window.show()
        #self.new_window.textBrowser.setText('1111')
        # if self.isPlaying:
        #     # 点击后暂停，设置图标为播放的图标
        #     self.actionread_books.setIcon(QtGui.QIcon('resources/customerservice.svg'))
        # else:
        #     # 点击后播放，设置图标为暂停图标
        #     self.actionread_books.setIcon(QtGui.QIcon('resources/pause.svg'))
        #     # 切换状态
        # self.isPlaying = not self.isPlaying

    # def read_books(self):

    # class AnotherWindow(QDialog, Ui_Dialog):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)

#class VoiceEngine():
    '''
    tts 语音工具类
    '''

    def __init__(self):
        '''
        初始化
        '''
        # tts对象
        self.__engine = pyttsx3.init()
        # 语速
        self.__rate = 150
        # 音量
        self.__volume = 50
        # 语音ID，0为中文，1为英文
        self.__voice = 0
        # 暂停标志



    @property
    def Rate(self):
        '''
        语速属性
        '''
        return self.__rate

    @Rate.setter
    def Rate(self, value):
        self.__rate = value

    @property
    def Volume(self):
        '''
        音量属性
        '''
        return self.__volume

    @Volume.setter
    def Volume(self, value):
        self.__volume = value

    @property
    def VoiceID(self):
        '''
        语音ID：0 -- 中文；1 -- 英文
        '''

        return self.__voice

    @VoiceID.setter
    def VoiceID(self, value):
        self.__voice = value

    def Say(self, text):
        '''
        播放语音
        '''
        self.__engine.setProperty('rate', self.__rate)
        self.__engine.setProperty('volume', self.__volume)
        voices = self.__engine.getProperty('voices')
        self.__engine.setProperty('voice', voices[self.__voice])

        self.__engine.say(text)
        self.__engine.runAndWait()
        #self.__engine.stop()

    def quit(self):
        self.__engine.stop()

class AnotherWindow(QDialog, Ui_Dialog):
    '''
    窗体类
    '''


    def __init__(self, task, parent=None):
        '''
        初始化窗体
        '''
        super(AnotherWindow, self).__init__(parent)
        self.setupUi(self)
        # self.init_info()
        #self.lines = lines
        #self.textBrowser.setText(self.lines[self.start])

        # 获取tts工具类实例
        self.stopSay = True
        self.running = True
        self.alive = False

        self.tasks = task
        self.audio_queue = Queue()
        self.engine = TTSEngine()
        self.player = player()

        
        #self.textEdit.insertPlainText('111')
        # self.verticalLayout_2.setRange(0, 300)
        # 设置初始文本




        # 进度条数据绑定到label中显示
        self.verticalSlider_2.valueChanged.connect(self.setRateTextValue)
        self.verticalSlider.valueChanged.connect(self.setVolumnTextValue)

        # 设置进度条初始值
        self.verticalSlider_2.setValue(self.engine.Rate)
        self.verticalSlider.setValue(self.engine.Volume)

        # 播放按钮点击事件
        self.pushButton_3.clicked.connect(self.onPlayButtonClick)

        #ComboBox选择事件
        self.comboBox.currentIndexChanged.connect(self.on_combo_box_changed)

    def refresh(self, text_read, start, end):
        self.lines = text_read
        self.start = start
        self.end = end

    # def init_info(self):
    #     self.setting = QtCore.QSettings("./config.ini", QtCore.QSettings.IniFormat)  # 配置文件
    #     self.setting.setIniCodec('utf-8')  # 设置配置文件的编码格式

    def setRateTextValue(self):
        '''
        修改语速label文本值
        '''
        value = self.verticalSlider_2.value()
        self.engine.Rate = value

    def setVolumnTextValue(self):
        '''
        修改音量label文本值
        '''
        value = self.verticalSlider.value()
        self.engine.Volume = value

    def on_combo_box_changed(self, index):
        if index == 0:
            self.engine.VoiceID = 0
        elif index == 1:
            self.engine.VoiceID = 1

    def playVoice(self):
        '''
        播放
        '''
        print('say.start')
        self.alive = True
        for i in range(self.start, self.end):
            self.line = self.lines[i].strip()

            #self.textBrowser.setText('1111')
            while self.stopSay:
                print('stopsay...')
                time.sleep(1.0)

            if not re.findall(r'^\s*?$', self.line):
                self.line = re.sub(r'\s', '', self.line)

                print('line=', i, self.line)
                #self.textEdit.clear()
                #self.deleteLastInsertedLine()
                self.textEdit.insertPlainText(self.line + "\n")
                #self.deleteLastInsertedLine()
                self.engine.Say(self.line)
            if not self.running:
                self.engine.quit()
                print('quit')
                return

            #self.textBrowser.clear()
            #self.textBrowser.setText(self.line)

            #text = self.line

            # self.t = threading.Thread(target=self.Say(text))
            # self.t.setDaemon(True)
            # self.t.start()


        print('Say.end')

    def onPlayButtonClick(self):
        '''
        播放按钮点击事件
        开启线程新线程播放语音，避免窗体因为语音播放而假卡死
        '''
        self.pause()
        #self.playVoice()
        #self.play_thread.start()
        if not self.alive:
            self.play_thread.start()
        #th.start_new_thread(self.playVoice, ())

    def pause(self):
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("play (1).svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("pause-one.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.stopSay = not self.stopSay
        if self.stopSay:
            self.pushButton_3.setIcon(icon1)
        else:
            self.pushButton_3.setIcon(icon2)



    def deleteLastInsertedLine(self):
        cursor = self.textEdit.textCursor()
        # 如果光标不在文本开头，则尝试删除上一行（因为我们刚刚插入了一行）
        if not cursor.atStart():
            cursor.movePosition(cursor.Up)  # 移动到上一行末尾
            cursor.movePosition(cursor.EndOfLine, cursor.KeepAnchor)  # 选择整行
            cursor.removeSelectedText()  # 删除选中的文本

    def closeEvent(self, event):
        result = QMessageBox.information(self, "退出听书", "退出后仍需等待当前行播放完毕", QMessageBox.Yes | QMessageBox.No,QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.running = False
            event.accept()
            QDialog.closeEvent(self, event)
        else:
            event.ignore()

        # self.running = False
        # event.accept()
        # QDialog.closeEvent(self, event)





if __name__ == '__main__':
    app = QApplication(sys.argv)  # 在 QApplication 方法中使用，创建应用程序对象
    myWin = MyMainWindow()  # 实例化 MyMainWindow 类，创建主窗口
    myWin.show()  # 在桌面显示控件 myWin
    sys.exit(app.exec_())  # 结束进程，退出程序