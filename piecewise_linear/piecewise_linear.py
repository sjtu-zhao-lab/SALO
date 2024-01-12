import numpy as np
import matplotlib.pyplot as plt
import pwlf

x = np.linspace(0.0, 1.0, 1000)
y = np.exp2(x)

my_pwlf = pwlf.PiecewiseLinFit(x, y)

my_pwlf.fit_with_breaks(np.array([0, 0.25, 0.5, 0.75, 1]))

xHat = np.linspace(0.0, 1.0, 10000)
yHat = my_pwlf.predict(xHat)

print(my_pwlf.slopes)
print(my_pwlf.intercepts)

plt.figure()
plt.plot(x, y, '-')
plt.plot(xHat, yHat, '-')
plt.savefig("./fit.png")