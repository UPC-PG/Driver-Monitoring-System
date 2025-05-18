from PyQt5 import uic
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.Qt import *
import headEst
import BlinkDetection
import YawnDetection
import HandsFreeDetection
import dataPack
import time
import imutils
from pyecharts import options as opts
from pyecharts.charts import Radar
import pyecharts.options as opts
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.charts import Bar
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot
from PyQt5.QtWebEngineWidgets import *
from LoginUI import *
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from qt_material import apply_stylesheet

class LoginWindow(QMainWindow):
#  此项目由1i9h7_b1u3,Vnhukvm,Syrena和bumianjun共同完成
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton.clicked.connect(self.go_to_inter)
        self.show()

    def go_to_inter(self):
        account = self.ui.lineEdit.text()
        password = self.ui.lineEdit_2.text()
        if account == "账号：123456" and password == '密码：123456':
            # 此处添加要跳转的窗口的类名
            self.close()
        else:
            pass

    #拖动
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.isMaximized() == False:
            self.m_flag = True
            self.m_Position = event.globalPos() - self.pos()  # 获取鼠标相对窗口的位置
            event.accept()
            self.setCursor(QtGui.QCursor(QtCore.Qt.OpenHandCursor))  # 更改鼠标图标

    def mouseNoveEvent(self, mouse_event):
        if QtCore.Qt.LeftButton and self.m_flag:
            self.move(mouse_event.globalPos() - self.m_Position)  # 更改窗口位置
            mouse_event.accept()

    def mouseReleaseEvent(self, mouse_event):
        self.m_flag = False
        self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

class MainUI:
    def __init__(self):
        # 装载UI文件
        self.ui = uic.loadUi('window.ui')
        # 播放器
        self.player = QMediaPlayer()
        self.player.setVideoOutput(self.ui.wgtPlayer)
        # 清空按钮
        self.ui.clearResult.clicked.connect(self.ui.textResult.clear)
        # 摄像头按钮 + 视频文件读取
        self.ui.btnCamOn.clicked.connect(self.camOn)
        self.ui.btnCamOff.clicked.connect(self.camOff)
        self.ui.btnSelect.clicked.connect(self.loadFile)
        # self.ui.btnPlayPause.clicked.connect(self.play)
        # self.ui.btnOutputResult.clicked.connect(self.output)
        # 勾选项目
        self.ui.ckbEye.clicked.connect(self.updateFunc)
        self.ui.ckbHands.clicked.connect(self.updateFunc)
        self.ui.ckbNod.clicked.connect(self.updateFunc)
        self.ui.ckbYawn.clicked.connect(self.updateFunc)
        # 摄像头初始化
        self.cap = cv2.VideoCapture()
        # 定时器
        self.camTimer = QTimer()
        self.drawTimer = QTimer()
        self.drawTimer.timeout.connect(self.drawSeries)
        self.camTimer.timeout.connect(self.logicProcess)
        # 图片自适应
        self.ui.lblCamera.setScaledContents(True)
        self.camFlag = False
        # 检测标识
        self.nodFlag = False
        self.blinkFlag = False
        self.yawnFlag = False
        self.handsFlag = False
        # 头部姿态（****）
        self.faceDetector = headEst.MyFaceDetector()
        self.headDetector = headEst.HeadPostEstimation(self.faceDetector)
        # 哈欠检测
        self.yawnDetector = YawnDetection.YawnDetect()
        # 闭眼检测
        self.blinkDetector = BlinkDetection.BlinkDetect()
        # 手部检测
        self.handsfreeDetector = HandsFreeDetection.HandsFreeDetection()
        # 数据包
        self.datapack = None
        # 雷达图
        # self.ui.radar.load(QUrl("D:/PythonProject/mainUI/radar.html"))
        # 面积图
        # self.ui.map.load(QUrl("D:/PythonProject/mainUI/map.html"))
        # self.ui.pie.load(QUrl("D:/PythonProject/mainUI/pie.html"))
        # self.ui.line.load(QUrl("D:/PythonProject/mainUI/line.html"))
        # self.ui.block.load(QUrl("D:/PythonProject/mainUI/block.html"))
        # 数据
        self.yawn = 0
        self.handsfree = 0
        self.nod = 0
        self.blink = 0
        self.keepNodding = False
        self.keepBlinking = False
        self.keepYawning = False
        self.keepFreeing = False
        self.yawnFrames = 0
        self.handsfreeFrames = 0
        self.nodFrames = 0
        self.blinkFrames = 0
        self.normalFrames = 0
        self.totalFrames = 0
        # 时间序列
        self.s1 = []
        self.s2 = []
        self.s3 = []
        self.s4 = []

    # 视频文件
    def loadFile(self):
        if self.camFlag:
            self.camOff()
        self.ui.wgtPlayer.raise_()
        fileUrl = QFileDialog.getOpenFileUrl()[0]
        self.cap = cv2.VideoCapture(fileUrl.toString())
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outVideo = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))
        self.camFlag = True
        cnt = 0

        while True:

            self.datapack = dataPack.driverData()
            flag, self.datapack.srcFrame = self.cap.read()
            if not flag:
                break
            self.datapack.srcFrame = cv2.flip(self.datapack.srcFrame, 1)
            self.datapack.dstFrame = self.datapack.srcFrame.copy()
            '''
            if self.nodFlag:
                self.datapack = self.headDetector.classify_pose_in_euler_angles(self.datapack)
                if self.datapack.isYawning:
                    self.ui.textResult.append('第', cnt, '帧' '打哈欠!')
                    self.yawn += 1
            if self.yawnFlag:
                self.datapack = self.yawnDetector.yawnDet(self.datapack)
                if self.datapack.isYawning:
                    self.ui.textResult.append('第', cnt, '帧', ' 打哈欠!')
                    self.yawn += 1
            if self.blinkFlag:
                self.datapack = self.blinkDetector.blinkDet(self.datapack)
                if self.datapack.isFreeing:
                    self.ui.textResult.append('第', cnt, '帧', ' 双手离开方向盘!')
                    self.handsfree += 1
            if self.handsFlag:
                self.datapack = self.handsfreeDetector.detect(self.datapack)
                if self.datapack.isFreeing:
                    self.ui.textResult.append('第', cnt, '帧', ' 双手离开方向盘!')
                    self.handsfree += 1
            '''
            if self.nodFlag:
                self.datapack = self.headDetector.classify_pose_in_euler_angles(self.datapack)
                if self.datapack.isNodding:
                    self.nodFrames += 1
                    if not self.keepNodding:
                        self.ui.textResult.append('第', str(cnt), '帧', ' 瞌睡点头!')
                        self.nod += 1

                self.keepNodding = self.datapack.isNodding
            if self.yawnFlag:
                self.datapack = self.yawnDetector.yawnDet(self.datapack)
                if self.datapack.isYawning:
                    self.yawnFrames += 1
                    if not self.keepYawning:
                        self.ui.textResult.append('第', str(cnt), '帧', ' 打哈欠!')
                        self.yawn += 1

                self.keepYawning = self.datapack.isYawning
            if self.blinkFlag:
                self.datapack = self.blinkDetector.blinkDet(self.datapack)
                if self.datapack.isBlinking:
                    self.blinkFrames += 1
                    if not self.keepBlinking:
                        self.ui.textResult.append('第', str(cnt), '帧', ' 瞌睡闭眼!')
                        self.blink += 1

                self.keepBlinking = self.datapack.isBlinking
            if self.handsFlag:
                self.datapack = self.handsfreeDetector.detect(self.datapack)
                if self.datapack.isFreeing:
                    self.handsfreeFrames += 1
                    if not self.keepFreeing:
                        self.ui.textResult.append('第', str(cnt), '帧', ' 双手离开方向盘!')
                        self.handsfree += 1

                self.keepFreeing = self.datapack.isFreeing

            self.datapack.timeStamp = time.time()
            if not self.datapack.isFreeing and \
                    not self.datapack.isYawning and \
                    not self.datapack.isNodding and \
                    not self.datapack.isBlinking:
                self.normalFrames += 1
            self.totalFrames += 1
            self.datapack.dstFrame = imutils.resize(self.datapack.dstFrame, width=width)
            outVideo.write(self.datapack.dstFrame)
            cnt += 1
            print("执行次数", cnt)
        outVideo.release()
        self.cap.release()

    # 摄像头
    def camOn(self):
        if self.cap.isOpened():
            self.camOff()
        self.ui.lblCamera.raise_()
        self.cap = cv2.VideoCapture(0)
        self.camFlag = True
        self.camTimer.start(40)
        # 10s更新一次图标
        # 单位 ms
        # *****
        self.drawTimer.start(10000)
        self.logicProcess()

    def logicProcess(self):
        self.datapack = dataPack.driverData()
        flag, self.datapack.srcFrame = self.cap.read()
        self.datapack.srcFrame = cv2.flip(self.datapack.srcFrame, 1)
        self.datapack.dstFrame = self.datapack.srcFrame.copy()
        if self.nodFlag:
            self.datapack = self.headDetector.classify_pose_in_euler_angles(self.datapack)
            if self.datapack.isNodding:
                self.nodFrames += 1
                if not self.keepNodding:
                    self.ui.textResult.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' 瞌睡点头!')
                    self.nod += 1

            self.keepNodding = self.datapack.isNodding
        if self.yawnFlag:
            self.datapack = self.yawnDetector.yawnDet(self.datapack)
            if self.datapack.isYawning:
                self.yawnFrames += 1
                if not self.keepYawning:
                    self.ui.textResult.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' 打哈欠!')
                    self.yawn += 1

            self.keepYawning = self.datapack.isYawning
        if self.blinkFlag:
            self.datapack = self.blinkDetector.blinkDet(self.datapack)
            if self.datapack.isBlinking:
                self.blinkFrames += 1
                if not self.keepBlinking:
                    self.ui.textResult.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' 瞌睡闭眼!')
                    self.blink += 1

            self.keepBlinking = self.datapack.isBlinking
        if self.handsFlag:
            self.datapack = self.handsfreeDetector.detect(self.datapack)
            if self.datapack.isFreeing:
                self.handsfreeFrames += 1
                if not self.keepFreeing:
                    self.ui.textResult.append(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ' 双手离开方向盘!')
                    self.handsfree += 1

            self.keepFreeing = self.datapack.isFreeing
        self.datapack.timeStamp = time.time()
        if not self.datapack.isFreeing and \
                not self.datapack.isYawning and \
                not self.datapack.isNodding and \
                not self.datapack.isBlinking:
            self.normalFrames += 1
        self.totalFrames += 1
        # self.datapack.isYawning and print('Yawn:', self.datapack.isYawning)
        # self.datapack.isFreeing and print('Hands Free:', self.datapack.isFreeing)
        # self.datapack.isNodding and print('Nod:', self.datapack.isNodding)
        # self.datapack.isBlinking and print('Blink:', self.datapack.isBlinking)
        width, height = self.datapack.dstFrame.shape[:2]
        qImg2Show = QtGui.QImage(self.datapack.dstFrame.data, height, width, QImage.Format_BGR888)
        self.ui.lblCamera.setPixmap(QPixmap.fromImage(qImg2Show))

    def camOff(self):
        self.camFlag = False
        self.camTimer.stop()
        self.cap.release()
        self.ui.lblCamera.clear()

    def updateFunc(self):
        self.nodFlag = self.ui.ckbNod.isChecked()
        self.blinkFlag = self.ui.ckbEye.isChecked()
        self.handsFlag = self.ui.ckbHands.isChecked()
        self.yawnFlag = self.ui.ckbYawn.isChecked()
        '''
        if self.ui.ckbNod.isChecked() == 0:
            self.nodFlag = False
        else:
            self.nodFlag = True

        if self.ui.ckbEye.isChecked() == 0:
            self.blinkFlag = False
        else:
            self.blinkFlag = True

        if self.ui.ckbYawn.isChecked() == 0:
            self.yawnFlag = False
        else:
            self.yawnFlag = True
        '''

    def drawSeries(self):
        self.drawRadar()
        self.drawPie()
        self.drawBar()
        self.drawMaps()
        # self.drawLine() 在 self.drawMaps() 调用
        # 绝对路径待查 ( ***** )
        self.ui.radar.load(QUrl("file:///radar.html"))
        self.ui.map.load(QUrl("file:///map.html"))
        self.ui.line.load(QUrl("file:///line.html"))
        self.ui.pie.load(QUrl("file:///pie.html"))
        self.ui.block.load(QUrl("file:///bar.html"))

    def drawRadar(self):
        tot = self.totalFrames
        v = [self.yawnFrames / tot, self.nodFrames / tot, self.handsfreeFrames / tot, self.blinkFrames / tot]
        maxV = max(v)
        v = [v]
        c = (
            Radar(init_opts=opts.InitOpts(width="441px", height="281px"))
                .add_schema(
                schema=[
                    opts.RadarIndicatorItem(name="打哈欠", max_=maxV, color='rgb(0, 0, 0)'),
                    opts.RadarIndicatorItem(name="瞌睡点头", max_=maxV, color='rgb(0, 0, 0)'),
                    opts.RadarIndicatorItem(name="双手离开方向盘", max_=maxV, color='rgb(0, 0, 0)'),
                    opts.RadarIndicatorItem(name="闭眼", max_=maxV, color='rgb(0, 0, 0)'),
                ],
                shape="circle",
                center=["50%", "50%"],
            )
                .add("", v,
                     color='rgb(255, 128, 0)',
                     areastyle_opts=opts.AreaStyleOpts(
                         color='rgba(255, 128, 0, 0.5)',
                         opacity=1
                     )
                     )
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
                .set_global_opts(
                legend_opts=opts.LegendOpts(selected_mode="single"),
                title_opts=opts.TitleOpts(title="危险驾驶行为雷达图"),
            )
                .render("radar.html")
        )

    def drawMaps(self):
        self.s1.append(self.yawnFrames)
        self.s2.append(self.nodFrames)
        self.s3.append(self.blinkFrames)
        if len(self.s1) > 6:
            self.s1.pop(0)
        if len(self.s2) > 6:
            self.s2.pop(0)
        if len(self.s3) > 6:
            self.s3.pop(0)
        x_data = ["1", "2", "3", "4", "5", "6"]
        v = []
        # 240 or self.totalFrames ( **** )
        # s1 = [i / self.totalFrames for i in self.s1]
        # s2 = [i / self.totalFrames for i in self.s2]
        # s3 = [i / self.totalFrames for i in self.s3]
        for i in range(len(self.s1)):
            v.append(self.s1[i] * 0.2 + self.s2[i] * 0.3 + self.s3[i] * 0.5)

        c = (
            Line(init_opts=opts.InitOpts(width="441px", height="281px"))
                .add_xaxis(xaxis_data=x_data)
                .add_yaxis(
                series_name="疲劳指数",
                y_axis=v,
                is_smooth=True,
                symbol="emptyCircle",
                is_symbol_show=True,
                label_opts=opts.LabelOpts(is_show=False),
                areastyle_opts=opts.AreaStyleOpts(opacity=1, color="#C67570"),
            )
                .set_global_opts(
                tooltip_opts=opts.TooltipOpts(is_show=False),
                yaxis_opts=opts.AxisOpts(
                    type_="value",
                    axistick_opts=opts.AxisTickOpts(is_show=True),
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                ),
                xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
                title_opts=opts.TitleOpts(title="疲劳指数面积图"),
            )
                .set_series_opts(
                label_opts=opts.LabelOpts(is_show=False),
                # MarkLineOpts：标记线配置项
                markline_opts=opts.MarkLineOpts(
                    # 标记线数据
                    data=[
                        # MarkLineItem：标记线数据项
                        opts.MarkLineItem(
                            name="疲劳指数阈值",
                            y=9,
                        )],
                    # 标签配置项，参考 `series_options.LabelOpts`
                    label_opts=opts.LabelOpts(),

                    # 标记线样式配置项，参考 `series_options.LineStyleOpts`
                    linestyle_opts=opts.LineStyleOpts(width=3, color='#0000FF', )
                )
            )
                .render("map.html")
        )
        self.drawLine()

    def drawPie(self):
        face = self.blinkFrames + self.yawnFrames
        head = self.nodFrames
        hand = self.handsfreeFrames
        normal = self.normalFrames

        close = self.blinkFrames
        yawn = self.yawnFrames
        nod = self.nodFrames

        inner_x_data = ["脸部", "头部", "手部", "正常"]
        inner_y_data = [face, head, hand, normal]
        inner_data_pair = [list(z) for z in zip(inner_x_data, inner_y_data)]

        outer_x_data = ["闭眼", "打哈欠", "点头", "手脱离方向盘", "正常"]
        outer_y_data = [close, yawn, nod, hand, normal]
        outer_data_pair = [list(z) for z in zip(outer_x_data, outer_y_data)]

        c = (
            Pie(init_opts=opts.InitOpts(width="441px", height="281px"))
                .add(
                series_name="帧数",
                data_pair=inner_data_pair,
                radius=[0, "30%"],
                label_opts=opts.LabelOpts(position="inner"),
            )
                .add(
                series_name="帧数",
                radius=["40%", "55%"],
                data_pair=outer_data_pair,
                label_opts=opts.LabelOpts(
                    position="outside",
                    formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
                    background_color="#eee",
                    border_color="#aaa",
                    border_width=1,
                    border_radius=4,
                    rich={
                        "a": {"color": "#999", "lineHeight": 15, "align": "center"},
                        "abg": {
                            "backgroundColor": "#e3e3e3",
                            "width": "100%",
                            "align": "right",
                            "height": 15,
                            "borderRadius": [4, 4, 0, 0],
                        },
                        "hr": {
                            "borderColor": "#aaa",
                            "width": "100%",
                            "borderWidth": 0.5,
                            "height": 0,
                        },
                        "b": {"fontSize": 12, "lineHeight": 22},
                        "per": {
                            "color": "#eee",
                            "backgroundColor": "#334455",
                            "padding": [2, 4],
                            "borderRadius": 2,
                        },
                    },
                ),
            )
                .set_global_opts(legend_opts=opts.LegendOpts(pos_left="left", orient="vertical"))
                .set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", formatter="{a} <br/>{b}: {c} ({d}%)"
                )
            )
                .render("pie.html")
        )

    def drawBar(self):
        close = self.blink
        yawn = self.yawn
        nod = self.nod
        hand = self.handsfree
        c = (
            Bar(init_opts=opts.InitOpts(width="441px", height="281px"))
                .add_dataset(
                source=[
                    ["score", "次数", "指标"],
                    [close / 10 * 100, close, "闭眼"],
                    [yawn / 3 * 100, yawn, "打哈欠"],
                    [nod / 5 * 100, nod, "点头"],
                    [hand / 3 * 100, hand, "手部"],
                ]
            )
                .add_yaxis(
                series_name="",
                y_axis=[],
                encode={"x": "次数", "y": "指标"},
                label_opts=opts.LabelOpts(position='right', is_show=True, vertical_align='middle'),
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="各指标次数条形图"),
                xaxis_opts=opts.AxisOpts(name="次数"),
                yaxis_opts=opts.AxisOpts(type_="category"),
                visualmap_opts=opts.VisualMapOpts(
                    orient="horizontal",
                    pos_left="center",
                    min_=1,
                    max_=100,
                    range_text=["High Score", "Low Score"],
                    dimension=0,
                    range_color=["#D7DA8B", "#E15457"],
                ),
            )
                .render("bar.html")
        )

    def drawLine(self):
        self.s4.append(self.handsfreeFrames)
        if len(self.s4) > 6:
            self.s4.pop(0)
        x_data = ["1", "2", "3", "4", "5", "6"]
        yawn_list = self.s1
        nod_list = self.s2
        close_list = self.s3
        hand_list = self.s4

        c = (
            Line(init_opts=opts.InitOpts(width="441px", height="281px"))
                .add_xaxis(xaxis_data=x_data)
                .add_yaxis(
                "闭眼",
                y_axis=close_list,
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                "打哈欠",
                y_axis=yawn_list,
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                "点头",
                y_axis=nod_list,
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .add_yaxis(
                "手脱离方向盘",
                y_axis=hand_list,
                linestyle_opts=opts.LineStyleOpts(width=2),
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title=""),
                xaxis_opts=opts.AxisOpts(name="时间"),
                yaxis_opts=opts.AxisOpts(
                    type_='value',
                    name="帧数",
                    splitline_opts=opts.SplitLineOpts(is_show=True),
                    is_scale=True,
                ),
            )
                .render("line.html")
        )


if __name__ == "__main__":
    appLogin = QApplication(sys.argv)
    winLogin = LoginWindow()
    appLogin.exec_()
    app = QApplication([])
    mainProgram = MainUI()
    extra = {

        # Button colors
        'danger': '#dc3545',
        'warning': '#ffc107',
        'success': '#17a2b8',

        # Font
        'font_family': 'Roboto',
    }
    apply_stylesheet(app, 'dark_cyan.xml', invert_secondary=False)
    mainProgram.ui.show()
    sys.exit(app.exec_())
