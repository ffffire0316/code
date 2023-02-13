"""
add noise signal by need

Author: Bruce Hou, Email: ecstayalive@163.com
"""
import numpy as np


def gen_gaussian_noise(self, signal, SNR):
    """
        :param signal: 原始信号
        :param SNR: 添加噪声的信噪比
        :return: 生成的噪声
        """
    P_signal = np.sum(abs(signal) ** 2) / len(signal)
    P_noise = P_signal / 10 ** (SNR / 10.0)
    return np.random.randn(len(signal)) * np.sqrt(P_noise)

