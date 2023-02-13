import json
import tkinter, tkinter.messagebox
from tkinter.scrolledtext import ScrolledText
import threading
from config import *
users=[]


class MainWindow:
    def __init__(self, send_call, username):
        self.send = send_call
        self.username = username

        # print("make a MainWindow")

    def main_window(self):
        global root1
        self.root1 = tkinter.Tk()
        self.root1.geometry("640x480")
        self.root1.title('XAI')
        self.root1.resizable(0, 0)

        # 消息界面
        self.listbox = ScrolledText(self.root1)
        self.listbox.place(x=5, y=0, width=640, height=320)
        self.listbox.tag_config('tag1', foreground='red', backgroun="yellow")
        self.listbox.insert(tkinter.END, '欢迎进入XAI群聊，大家开始聊天吧!', 'tag1')

        # 消息输入框
        self.INPUT = tkinter.StringVar()
        self.INPUT.set('')
        entryIuput = tkinter.Entry(self.root1, width=120, textvariable=self.INPUT)
        entryIuput.place(x=5, y=320, width=580, height=170)

        # 在线用户列表
        self.listbox1 = tkinter.Listbox(self.root1)
        self.listbox1.place(x=510, y=0, width=130, height=320)

        # 消息发送键
        sendButton = tkinter.Button(self.root1, text="\n发\n\n\n送", anchor='n', command=self.send,
                                    font=('Helvetica', 18),
                                    bg='white')
        sendButton.place(x=585, y=320, width=55, height=300)
        self.root1.bind('<Return>', self.send)
        self.root1.mainloop()

    def show_message(self, data):
        print("in the show_message data:",data,type(data))
        # global users
        global uses
        # uses = []

        try:
            print("进入用户更新")
            print(type(data))
            uses = json.loads(data)
            self.listbox1.delete(0, tkinter.END)
            self.listbox1.insert(tkinter.END, "当前用户")
            self.listbox1.insert(tkinter.END, "--------group chat-----")
            for x in range(len(uses)):
                self.listbox1.insert(tkinter.END, uses[x])
                users.append("-------group chat-----")
        except:
            print("进入消息更新")
            data = data.split('~')
            message = data[0]
            userName = data[1]
            chatwith = data[2]
            message = '\n' + message
            if chatwith == '------Group chat-------':  # 群聊
                if userName == self.username:
                    self.listbox.insert(tkinter.END, message)
                else:
                    self.listbox.insert(tkinter.END, message)
            elif userName == self.username or chatwith == self.username:  # 私聊
                if userName == self.username:
                    self.listbox.tag_config('tag2', foreground='red')
                    self.listbox.insert(tkinter.END, message, 'tag2')
                else:
                    self.listbox.tag_config('tag3', foreground='green')
                    self.listbox.insert(tkinter.END, message, 'tag3')

            self.listbox.see(tkinter.END)

    def get_input(self):
        # print("in the get_put1:", self.INPUT.get())
        data = self.INPUT.get()
        # data = self.INPUT.get()+ '~' + user + '~' + chat
        self.INPUT.set('')
        return data
        # return self.INPUT.get()
        # return self.INPUT.get()

# a = MainWindow(send_test)
# a.main_window()
