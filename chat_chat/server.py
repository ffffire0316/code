import socket
# import eventlet.wsgi
import logging
import random
import threading
import os, sys
import os.path
import json
import queue

hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)
logging.basicConfig()
logging.warning("server ip:" + ip)

users = []  # 0:userName 1:connection
messages = queue.Queue()
lock = threading.Lock()


def number_list():
    numbers = []
    for i in range(len(users)):
        numbers.append(users[i][0])
    return numbers


class Server(threading.Thread):
    global users, que, lock

    def __init__(self):
        threading.Thread.__init__(self)
        # socket.AF_INET ipv4的版本
        # socket.SOCK_STREAM 协议
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # buy the iphone
        os.chdir(sys.path[0])

    # 先是接受用户名，如果不存在则用ip和port代替 重复则加上后缀
    def receive(self, conn, addr):
        user = conn.recv(1024)
        user = user.decode()
        print("server have recvive the user: ",user)
        if user == 'user don not  exist':
            user = addr[0] + ':' + str(addr[1])
        tag = 1
        temp = user
        for i in range(len(users)):
            if user[i][0] == user:
                tag = tag + 1
                user = temp + str(tag)
        users.append((user, conn))

        USERS = number_list()
        self.load(USERS, conn)

        try:  # 获取用户名后边不断接受信息
            while True:
                message = conn.recv(1024)
                message = message.decode()
                message = user + ':' + message
                print("server receive message：",message)
                self.load(message, addr)
                # 如果用户断开连接，将该用户从用户列表中删除，然后更新用户列表。
        except:
            j = 0  # 用户断开连接
            for man in users:
                if man[0] == user:
                    users.pop(j)  # 服务器段删除退出的用户
                    break
                j = j + 1

                USERS = number_list()  # 重新统计在线人数
                self.load(USERS, addr)
                conn.close()

    def load(self, data, addr):
        lock.acquire()
        try:
            messages.put((addr, data))
        finally:
            lock.release()

    def SendData(self):
        while True:
            if not messages.empty():
                message = messages.get()
                # message[1]是导入的数据  [0]是来源的地址（ip+port）第二次以后接受的数据 self.load(message,addr)
                # message【1】为str 是发送的消息内容
                # print("message[0]:", message[0])
                # print("message[1]:", message[1])
                # print("len(users):", len(users))
                if isinstance(message[1], str):
                    for i in range(len(users)):
                        data = ' ' + message[1]
                        # uesrs[i][1]表示第i个用户 [1]表示conn
                        users[i][1].send(data.encode())
                        print('server have sent the data:',data)
                        print('\n')
                # 第一次接受的message是（self.load(USERS, conn)），message【1】是USERS【list】类型
                if isinstance(message[1], list):
                    data = json.dumps(message[1])
                    for i in range(len(users)):
                        try:
                            users[i][1].send(data.encode())
                            print("server have sent the user name",data)
                        except:
                            pass

    def run(self):
        # 绑定ip和端口
        self.s.bind(("127.0.0.1", 7080))
        self.s.listen(5)
        print('start.....')
        q = threading.Thread(target=self.SendData)
        q.start()
        while True:
            conn, addr = self.s.accept()
            print("conn", conn)
            print('client addr', addr)
            t = threading.Thread(target=self.receive, args=(conn, addr))
            t.start()
            # print("rec_data:",rec_data)
        self.s.close()


if __name__ == '__main__':
    cserver = Server()
    cserver.start()
# print(s)
