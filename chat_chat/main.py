import socket
import tkinter
import tkinter.messagebox
import threading
import json
import tkinter.filedialog
from tkinter.scrolledtext import ScrolledText
import tkinter, tkinter.messagebox
import time
from login_window import LoginWindow
from main_window import MainWindow
from chat_client import ChatClient
# from config import *
chat='------Group chat-------'  # 聊天对象



def handding_login(*args):
    # 调用chat_login_pannel模块的对象实例方法获取输入的用户名和密码
    # global root0
    print("in the handding")
    global IP,PROT,user
    IP, PROT, user = root0.get_input()
    if not user:
        tkinter.messagebox.showwarning('warning', message='用户名为空!')
    else:
        root0.root0.destroy()  # 对象调用实例方法关闭登陆界面

    global client
    client = ChatClient(IP,PROT)
    client.login_send(user)

    global root1
    root1 = MainWindow(send_test,user)

    threading.Thread(target=recv_data).start()

    root1.main_window()

def recv_data():
    #   获取client收到的数据
    time.sleep(0.5)

    print("in the thread :")
    while True:
        try:
            data = client.receive()
            print("in recv:",data,type(data))
            # 将消息显示至消息框
            root1.show_message(data)

        except Exception as e:
            print("error appear"+str(e))
            break

def send_test(*args):
    data=root1.get_input()
    global user
    global client
    message = data + '~' + user + '~' + chat
    #  客户端发送数据给server
    print("root1发送给server数据：",message)
    client.message_send(message)
    print("in the send_test data:", data)
    print("in the send_test user:",user)
    print("-------------------------")



# def go_to_main_window():


if __name__ == "__main__":
    global root0

    root0 = LoginWindow(handding_login)
    root0.login_window()
