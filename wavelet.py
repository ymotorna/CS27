import numpy as np
import matplotlib.pyplot as plt
import pywt

N = 64
# I = [0; 10]
step = 10/N


# f(t) function
def f(t):
    if t >= 0 and t <= 4:
        return 0.25 * (1 - abs(t-2)/2)
    else:
        return 0

f_values = [f(t) for t in np.arange(0, 10, 0.01)]
# discrete points
x_N = [f(t) for t in np.arange(0, 10, step)]


# dwt
cA_haar, cD_haar = pywt.dwt(x_N, 'haar')
cA_coiflet, cD_coiflet = pywt.dwt(x_N, 'coif1')

# idwt haar
haar_cA_only = pywt.idwt(cA_haar, [0]*len(cD_haar), 'haar')
haar_cD_only = pywt.idwt([0]*len(cA_haar), cD_haar, 'haar')
haar_whole = pywt.idwt(cA_haar, cD_haar, 'haar')


# idwt coiflet
coiflet_cA_only = pywt.idwt(cA_coiflet, [0]*len(cD_coiflet), 'coif1')
coiflet_cD_only = pywt.idwt([0]*len(cA_coiflet), cD_coiflet, 'coif1')
coiflet_whole = pywt.idwt(cA_coiflet, cD_coiflet, 'coif1')


# plot
fig, ax = plt.subplots(2, 2, figsize=(15, 10), tight_layout=True)

x_discrete_values = np.arange(0, 10, step)
x_dwt_haar_values = np.linspace(0, 10, len(cA_haar))
x_dwt_coiflet_values = np.linspace(0, 10, len(cA_coiflet))

ax[0][0].grid()
ax[0][0].plot(np.arange(0, 10, 0.01), f_values, label='f(x)')
ax[0][0].scatter(x_dwt_haar_values, cA_haar, label='cA Haar')
ax[0][0].scatter(x_dwt_haar_values, cD_haar, label='cD Haar')
ax[0][0].set_title('Haar')
ax[0][0].legend()

ax[0][1].grid()
ax[0][1].plot(np.arange(0, 10, 0.01), f_values, label='f(x)')
ax[0][1].plot(x_discrete_values, haar_cA_only, label='cA idwt Haar')
ax[0][1].plot(x_discrete_values, haar_cD_only, label='cD idwt Haar')
ax[0][1].plot(x_discrete_values, haar_whole, label='whole idwt Haar')
ax[0][1].set_title('idwt Haar')
ax[0][1].legend()

ax[1][0].grid()
ax[1][0].plot(np.arange(0, 10, 0.01), f_values, label='f(x)')
ax[1][0].scatter(x_dwt_coiflet_values, cA_coiflet, label='cA Coiflet')
ax[1][0].scatter(x_dwt_coiflet_values, cD_coiflet, label='cD Coiflit')
ax[1][0].set_title('Coiflet')
ax[1][0].legend()

ax[1][1].grid()
ax[1][1].plot(np.arange(0, 10, 0.01), f_values, label='f(x)')
ax[1][1].plot(x_discrete_values, coiflet_cA_only, label='cA idwt Coiflet')
ax[1][1].plot(x_discrete_values, coiflet_cD_only, label='cD idwt Coiflit')
ax[1][1].plot(x_discrete_values, coiflet_whole, label='whole idwt Coiflet')
ax[1][1].set_title('idwt Coiflet')
ax[1][1].legend()

plt.show()





