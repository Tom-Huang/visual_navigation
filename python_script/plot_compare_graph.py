import numpy as np
import matplotlib.pyplot as plt

# empty cells percentage
x = [0.35,0.37,0.4,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.50]

# rmse
y = [0.1303,0.3459,0.1295,0.1099,0.1116,0.1563,0.1503,0.1304,0.1268,0.1433,0.9060,0.4288]

# optimization time
z = [430.423,400.085,333.723,305.043,307.711,301.82,274.728,268.572,291.259,242.226,248.023,244.352]

# detection time
w = [3.03643,2.74359,2.73232,2.43286,2.26378,2.62328,2.52254,2.26326,2.7443,2.42524,2.23357,2.42336]
x_ex5 = [0.35,0.5]
w_ex5 = [44.0214,44.0214]

# f2f optical flow time
k = [143.192,128.286,133.53,117.438,120.597,135.881,127.484,120.76,128.453,118.085,118.31,133.065]

fig1 = plt.figure(1)
fig1.suptitle("empty cells percentage vs rmse")
plt.plot(x,y,"bo-")
plt.xlabel('empty cells percentage')
plt.ylabel('rmse')

fig2 = plt.figure(2)
fig2.suptitle("empty cells percentage vs optimization time")
plt.plot(x,z,"bo-")
plt.xlabel('empty cells percentage')
plt.ylabel('optimization time/second')


fig3 = plt.figure(3)
fig3.suptitle("empty cells percentage vs detection time")
plt.plot(x,w,"bo-")
plt.plot(x_ex5,w_ex5,"r-")
plt.xlabel('empty cells percentage')
plt.ylabel('detection time/second')
#plt.ylim(0.10,0.16)

fig4 = plt.figure(4)
fig4.suptitle("empty cells percentage vs f2f optical flowtime")
plt.plot(x,k,"bo-")
plt.xlabel('empty cells percentage')
plt.ylabel('f2f optical flow time/second')


plt.show()
