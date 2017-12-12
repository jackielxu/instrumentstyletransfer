import numpy as np
import matplotlib.pyplot as plt


default = [
0.00072526053,
0.002012569,
0.0054193819,
0.016593169,
0.0040836502,
0.012274484
]
default = [(i/100.) for i in default]


# Varying hidden units:
hu1 = [
6.4748207e-13,
1.1114581e-12,
4.4176002e-10,
2.4532734e-09,
1.8337033e-10,
2.0677349e-12
]
hu1 = [(i) for i in hu1]

hu500 = [
0.031839117,
0.045599978,
0.12239362,
0.41998094,
0.17756835,
0.2376343
]
hu500 = [(i/500.) for i in hu500]


# Varying forget bias
fb0 = [
0.00019810173,
0.00028306374,
0.00069117639,
0.0044512777,
0.00072917517,
0.0049024299
]
fb0 = [(i/100.) for i in fb0]

fb5 = [
1.1284686,
1.1515893,
1.159517,
127.31981,
256.20889,
150.29135
]
fb5 = [(i/100.) for i in fb5]


# Randomized spectrogram initialization
rand = [
7.9125614,
4.8903146,
7.9664559,
4.5534515,
4.3076162,
8.8386412
]
rand = [(i/100.) for i in rand]

# data to plot
n_groups = 6

## Hidden Units
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, hu1, bar_width,
                 alpha=opacity,
                 label='1 HU')


rects2 = plt.bar(index + bar_width, default, bar_width,
                 alpha=opacity,
                 label='100 HU')

rects3 = plt.bar(index + 2*bar_width , hu500, bar_width,
                 alpha=opacity,
                 label='500 HU')

plt.yscale('log')
plt.xlabel('Style Content Pair')
plt.ylabel('Loss (Log scale)')
plt.title('RNN Model Loss Comparisons for Varying Hidden Units')
plt.xticks(index + bar_width, ('SPCC', 'SVCC', 'SPCV', 'SCCP', 'SCCV', 'SVCP'))
plt.legend()
plt.tight_layout()
plt.savefig('hu_loss.png')
plt.show()


## Forget bias
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index , fb0, bar_width,
                 alpha=opacity,
                 label='Forget bias 0')

rects2 = plt.bar(index + bar_width, default, bar_width,
                 alpha=opacity,
                 label='Forget bias 1')

rects3 = plt.bar(index + 2*bar_width , fb5, bar_width,
                 alpha=opacity,
                 label='Forget bias 5')

plt.yscale('log')
plt.xlabel('Style Content Pair')
plt.ylabel('Loss (Log scale)')
plt.title('RNN Model Loss Comparisons for Varying Forget Bias')
plt.xticks(index + bar_width, ('SPCC', 'SVCC', 'SPCV', 'SCCP', 'SCCV', 'SVCP'))
plt.legend()
plt.tight_layout()
plt.savefig('fb_loss.png')
plt.show()



## Initialization
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index , default, bar_width,
                 alpha=opacity,
                 label='Init with content spectrogram')

rects6 = plt.bar(index + bar_width , rand, bar_width,
                 alpha=opacity,
                 label='Init with random spectrogram')

plt.yscale('log')
plt.xlabel('Style Content Pair')
plt.ylabel('Loss (Log scale)')
plt.title('RNN Model Loss Comparisons for Varying Output Initialization')
plt.xticks(index + bar_width, ('SPCC', 'SVCC', 'SPCV', 'SCCP', 'SCCV', 'SVCP'))
plt.legend()
plt.tight_layout()
plt.savefig('init_loss.png')
plt.show()
