import socket
import tkinter
import tkinter.messagebox
import threading
import json
import tkinter.filedialog
from tkinter.scrolledtext import ScrolledText
from login_window import LoginWindow
from main_window import MainWindow
from config import *


class ChatClient(threading.Thread):
    # 构造方法
    def __init__(self,ip,port):
        print("初始化chatclient")
        # 创建对象的同时，会创建连接服务器的socket
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建socket
        self.client_socket.connect((ip, int(port)))  # 请求连接服务器
        self.ip=ip
        self.port=int(port)

        # r=threading.Thread(target=self.receive)
        # r.start()

    def login_send(self,user):
#   root0的消息user发送server

        if user:
            self.client_socket.send(user.encode())
            print(type(user.encode()))
        else:
            self.client_socket.send('用户名不存在'.encode())
            self.user = self.ip + ':' + str(self.port)

    def message_send(self,message):

        # print("in the message_send;", message)
        self.client_socket.send(message.encode())

    def receive(self):
        while True:
            print("in the receive funciton")
            data=self.client_socket.recv(1024)
            data=data.decode()

            # uses = json.loads(data)
            # print(uses,type(uses))
            return data





# user="fire"
# c=ChatClient()
# c.message_send(user)

