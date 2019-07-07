from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import csv



fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')

data = csv.reader(open('groundtruth_cleaned_full.csv'))

d= list(data)
d = np.array(d)

x = d[:,1]
y = d[:,2]
z = d[:,3]

x = x.astype(np.float)
y = y.astype(np.float)
z = z.astype(np.float)


ax1.scatter(x,y,z)
plt.show()
