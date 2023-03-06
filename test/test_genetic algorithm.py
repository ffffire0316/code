import matplotlib.pyplot as plt
import numpy as np

x_value = np.random.randint(140, 180, 200)

plt.hist(x_value, bins=10)

plt.title("data analyze")
plt.xlabel("height")
plt.ylabel("rate")

plt.show(block=True)

