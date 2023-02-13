import numpy as np
import matplotlib.pyplot as plt
#
# 参数定义 可参考
# 平衡电位
E_Na = 115.0  # [mV]
E_K = -12.0  # [mV]
E_L = 10.6  # [mV]

# 最大电导
g_Na = 120.0  # [mS]
g_K = 36.0  # [mS]
g_L = 0.3  # [mS]

# 仿真时间
dt = 0.01  # [ms]
T = 40  # [ms]
t = np.arange(0, T, dt)
# 各参数初始化
V = np.zeros(len(t))
n = np.zeros(len(t))
m = np.zeros(len(t))
h = np.zeros(len(t))
I_E=np.zeros(len(t))
# I_E = 0.0
V[0] = 0.0
# 初始化的h，m，n的值
h[0] = 0.59
m[0] = 0.05
n[0] = 0.31

# 在10ms是注入10mA的电流，然后15ms再次置电流为0
for i in range(1, len(t)):
    # if i == 1000:
    #     I_E[i] = 1.0
    # if i == 1100:
    #     I_E[i] = 0.0
    # if i == 1200:
    #     I_E[i] = 3.0
    # if i == 1700:
    #     I_E[i] = 0.0
    if  1000<i<1500:
        I_E[i] = 3.0
    if  2400<i<2900:
        I_E[i] = 5.0
    # if  2600<i<3100:
    #     I_E[i] = 3.0

    # Calculate the alpha and beta functions
    alpha_n = (0.1 - 0.01 * V[i - 1]) / (np.exp(1 - 0.1 * V[i - 1]) - 1)
    alpha_m = (2.5 - 0.1 * V[i - 1]) / (np.exp(2.5 - 0.1 * V[i - 1]) - 1)
    alpha_h = 0.07 * np.exp(-V[i - 1] / 20.0)

    beta_n = 0.125 * np.exp(-V[i - 1] / 80.0)
    beta_m = 4.0 * np.exp(-V[i - 1] / 18.0)
    beta_h = 1 / (np.exp(3 - 0.1 * V[i - 1]) + 1)

    # Calculate the time constants and steady state values
    tau_n = 1.0 / (alpha_n + beta_n)
    inf_n = alpha_n * tau_n

    tau_m = 1.0 / (alpha_m + beta_m)
    inf_m = alpha_m * tau_m

    tau_h = 1.0 / (alpha_h + beta_h)
    inf_h = alpha_h * tau_h

    # 更新n,m,h
    n[i] = (1 - dt / tau_n) * n[i - 1] + (dt / tau_n) * inf_n
    m[i] = (1 - dt / tau_m) * m[i - 1] + (dt / tau_m) * inf_m
    h[i] = (1 - dt / tau_h) * h[i - 1] + (dt / tau_h) * inf_h

    # 更新膜电位方程
    I_Na = g_Na * (m[i] ** 3) * h[i] * (V[i - 1] - E_Na)
    I_K = g_K * (n[i] ** 4) * (V[i - 1] - E_K)
    I_L = g_L * (V[i - 1] - E_L)


    dv = I_E[i] - (I_Na + I_K + I_L)
    V[i] = V[i - 1] + dv * dt

plt.clf()
plt.subplot(2, 3, 1)
plt.plot(t, V)
# 膜电位变化
plt.title('Membrane potential')
# plt.subplot(2, 3, 2)
# plt.plot(t, n)
# plt.plot(t, m)
# plt.plot(t, h)
# plt.title('Gating variables')
# plt.legend(['n', 'm', 'h'])
# plt.subplot(2, 3, 3)
# plt.plot(t, g_Na * h * m ** 3)
# plt.plot(t, g_K * n ** 4)
# plt.title('Ionic currents')
# plt.legend(['Na', 'K'])


plt.subplot(2,3,4)
plt.plot(t,I_E)
plt.title('activate I')

plt.show()
