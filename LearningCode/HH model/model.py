import numpy as np
import matplotlib.pyplot as plt

#1.set the Membrane potential -80mv~80mv
V=np.arange(-150,100,0.01)
alpha_n=0.01*(V+55)/(1-np.exp(-(V+55)/10))
alpha_m=0.01*(V+40)/(1-np.exp(-(V+40)/10))
alpha_h=0.07*np.exp(-(V+65)/20)

beta_n=0.125*np.exp(-(V+65)/80)
beta_m=4*np.exp(-(V+65)/18)
beta_h=1/(1+np.exp(-(V+35)/10))

# alpha_n = 0.01 *(10-V)/(np.exp((10-V)/10)-1)
# alpha_m = 0.1*(25-V)/((np.exp((25-V)/10))-1)
# alpha_h = 0.07 * np.exp(-(V/20))
#
# beta_m = 4* (np.exp(-V/18))
# beta_h = 1/(np.exp((30-V)/10)+1)
# beta_n = 0.125 * np.exp(-(V/80))


#概率
inf_n=alpha_n/(alpha_n+beta_n)
inf_m=alpha_m/(alpha_m+beta_m)
inf_h=alpha_h/(alpha_h+beta_h)

tau_n=1/(alpha_n+beta_n)
tau_m=1/(alpha_m+beta_m)
tau_h=1/(alpha_h+beta_h)

#绘图
plt.clf()
plt.subplot(1,2,1)
plt.plot(V,inf_n)
plt.plot(V,inf_m)
plt.plot(V,inf_h)
plt.title('Steady state values')
plt.xlabel('Voltage (mV)')
plt.subplot(1,2,2)
plt.plot(V,tau_n)
plt.plot(V,tau_m)
plt.plot(V,tau_h)
plt.title('Time constants')
plt.xlabel('Voltage (mV)')
plt.legend(['n','m','h'])
plt.show()
