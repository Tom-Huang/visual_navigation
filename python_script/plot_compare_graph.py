import numpy as np
import matplotlib.pyplot as plt

x = [0.35,0.37,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50]
y = [0.1303,0.3459,0.1295,0.1099,0.1116,0.1563,0.1503,0.1304,0.1268,0.1433,0.9060,0.8189]

fig = plt.figure()
fig.suptitle("empty cells percentage vs rmse")
plt.plot(x,y,"bo-")

plt.xlabel('empty cells percentage')
plt.ylabel('rmse')
#plt.ylim(0.10,0.16)


plt.show()
