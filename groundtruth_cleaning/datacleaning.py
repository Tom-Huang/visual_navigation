import csv
import numpy as np


data_estimate = csv.reader(open('data_estimate.csv'))
data_gt = csv.reader(open('data_gt.csv'))

e = list(data_estimate)
e = np.array(e)
g = list(data_gt)
g = np.array(g)

#time step list for estimated poses
e_pure = e[:,0]
e_pure = np.delete(e_pure, 0)

#time step list for ground truth
g_pure = g[:,0]
g_pure = np.delete(g_pure, 0)


writer = csv.writer(open('output.csv', 'w'))
writer_match = csv.writer(open('match.csv', 'w'))
match = []

for i in range(e_pure.shape[0]): # estimate: around 2900 pictures
    for j in range(g_pure.shape[0]): # ground truth: many 
        if e_pure[i] == g_pure[j]:
            writer.writerow(g[j]) 
            match.append([i,j])

writer_match.writerows(match)

#writer.writerow(g[0])


#print(g[0])


