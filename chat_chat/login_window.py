from tkinter import *  # 导入模块，用户创建GUI界面
import tkinter, tkinter.messagebox
import tkinter.font as tf  # 处理字体样式和颜色的类
import time
# import chat_mysql  # 导入处理mysql的模块
from PIL import Image  # 导入处理图像模块


class LoginWindow:
    def __init__(self, handle_login):
        # 初始化参数实例变量
        # self.user_name = user_name
        # self.send_message = send_message
        # self.send_mark = send_mark
        # self.refurbish_user = refurbish_user
        # self.private_talk = private_talk
        # self.close_main_window = close_main_window
        self.Login = handle_login

    def login_window(self):
        global root0
        self.root0 = tkinter.Tk()
        self.root0.geometry("300x150")
        self.root0.title('用户登陆窗口')
        self.root0.resizable(0, 0)
        one = tkinter.Label(self.root0, width=300, height=150, bg="LightBlue")
        one.pack()

        # 声明用户名密码变量
        self.IP0 = tkinter.StringVar()
        self.IP0.set('')
        self.USER = tkinter.StringVar()
        self.USER.set('')
        # 文本标签和位置
        labelIP = tkinter.Label(self.root0, text='IP地址', bg="LightBlue")
        labelIP.place(x=20, y=20, width=100, height=40)
        labelUSER = tkinter.Label(self.root0, text='用户名', bg="LightBlue")
        labelUSER.place(x=20, y=70, width=100, height=40)
        # 输入框标签和位置
        entryIP = tkinter.Entry(self.root0, width=60, textvariable=self.IP0)
        entryIP.place(x=120, y=25, width=100, height=30)
        entryUSER = tkinter.Entry(self.root0, width=60, textvariable=self.USER)
        entryUSER.place(x=120, y=75, width=100, height=30)
        print("before")
        # 设置登录按钮及位置，按钮事件为handle_login函数
        loginButton = tkinter.Button(self.root0, text="登录", command=self.Login, bg="Yellow")
        loginButton.place(x=135, y=110, width=40, height=25)
        self.root0.bind('<Return>', self.Login)
        self.root0.mainloop()

    def get_input(self):

        print("in the get_put:", self.IP0.get(), self.USER.get())
        if self.IP0.get()=='0':
            IP, PORT ="127.0.0.1",'7080'
            user = self.USER.get()

        else:
            IP, PORT = self.IP0.get().split(':')
            # return self.IP0.get(),self.USER.get()

        return IP, PORT, user

# a = MainWindow(Login)
# a.login_window()
