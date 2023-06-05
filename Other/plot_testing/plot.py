import numpy as np
import matplotlib.style as mplstyle
import matplotlib.pyplot as plt
from datetime import datetime


mplstyle.use("fast")
plt.ioff()


times = []
n = 100


for i in range(n):

    x = np.linspace(-1, 1, num=100)
    coeffs = np.random.rand(10)
    y = np.polyval(coeffs, x)

    start_time = datetime.now()

    plt.plot(x, y)
    plt.savefig(f"plots/plot{i}.png")
    plt.clf()

    end_time = datetime.now()

    times.append((end_time - start_time).total_seconds())

print(f"Average Time: {sum(times)} for {n}")
