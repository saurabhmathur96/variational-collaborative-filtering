import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import numpy as np

import json

data = '''
4.8287 4.8545
2.2277 3.2117
0.9175 1.0682
0.8579 0.9669
0.8218 0.9523
0.7723 0.9481
0.7298 0.9525
0.6990 0.9610
0.6789 0.9697
0.6663 0.9769'''
rows = (row.split() for row in data.strip().splitlines())
rows = [(float(train_error), float(test_error)) for train_error, test_error in rows]
train_rmse, test_rmse = zip(*rows)



plt.figure(figsize=(10, 6)) 

tableau20blind = [(0, 107, 164), (255, 128, 14), (171, 171, 171), (89, 89, 89),
             (95, 158, 209), (200, 82, 0), (137, 137, 137), (163, 200, 236),
             (255, 188, 121), (207, 207, 207)]
for i in range(len(tableau20blind)):
    r, g, b = tableau20blind[i]  
    tableau20blind[i] = (r / 255., g / 255., b / 255.)

x = list(range(1, len(train_rmse)+1))

y1 = train_rmse
y2 = test_rmse


plt.plot(x, y1, color=tableau20blind[0], label='Insertion sort', lw=2) # Blue
plt.plot(x, y2, color=tableau20blind[1], label='Merge sort', lw=2) # Red

plt.xticks(x, fontsize=14)

plt.yticks(fontsize=14)
plt.ylim(1e-5, np.max([*y1, *y2])+1e-5)
for y in plt.yticks()[0]:    
    plt.plot(x, [y] * len(x), "--", lw=0.5, color=tableau20blind[3])
    




plt.text(x[-1] + np.mean(x)*.01, y1[-1], 'Train RMSE', color=tableau20blind[0], fontsize=14)
plt.text(x[-1] + np.mean(x)*.01, y2[-1], 'Test RMSE', color=tableau20blind[1], fontsize=14)
#plt.text(x[-1] + np.mean(x)*.01, y3[-1], 'Quick sort', color=tableau20blind[5], fontsize=14)



plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: format(int(x), ',')))


plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.setp(plt.gca().get_xticklabels(), color=tableau20blind[3]) # gray
plt.setp(plt.gca().get_yticklabels(), color=tableau20blind[3])
plt.gca().tick_params(axis='both', length=0)
plt.tight_layout(pad=5)

plt.xlabel('Iterations', color=tableau20blind[3], fontsize=14)

plt.savefig('ml-100k-rank-10.png')
