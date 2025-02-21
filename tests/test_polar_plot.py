import numpy as np
import matplotlib.pyplot as plt

r = np.geomspace(1, 100, num=10)
th = np.linspace(0, 360, num=50)
th = np.radians(th)
v = np.random.rand(len(r), len(th))

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection' :'polar'}, figsize=(10, 5))
ax1.set_rlim(1, 100)
print(th.shape)
print(r.shape)
print(v.shape)
ax1.pcolormesh(th, r, v)

ax2.set_rlim(1, 100)
ax2.set_rscale('log')
ax2.pcolormesh(th, r, v)
plt.show()
