import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

N = 64
# I = [0; 10]
step = 10/N
fs = 1/step    # частота дискретизації

# f(t) function
def f(t):
    if t >= 0 and t <= 4:
        return 0.25 * (1 - abs(t-2)/2)
    else:
        return 0

# x[N]
x_N = [f(t) for t in np.arange(0, 10, step)]

f_values = [f(t) for t in np.arange(0, 10, 0.01)]

# graph
fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

ax[0].grid()
ax[0].plot(np.arange(0, 10, 0.01), f_values)
ax[0].set_title('Continuous signal')
ax[0].set_xlabel('t')
ax[0].set_ylabel('f(t)')

ax[1].grid()
ax[1].scatter(np.arange(0, 10, step), x_N)
ax[1].set_title('Discrete signal')
ax[1].set_xlabel('t')
ax[1].set_ylabel('x(t)')

plt.tight_layout()
plt.show()


def fourier_transform(freq):
    result = sp.integrate.quad(lambda t: f(t) * np.exp(-2j * np.pi * t * freq), -np.inf, np.inf)[0]
    return result


def discrete_transform(x_val, k):
    Xk = 0
    for n in range(N):
        Xk += x_val[n] * np.exp(-2j * k * n * np.pi / N)
    return step*Xk


# частотний спектр
Xk_values = [discrete_transform(x_N, k) for k in range(N)]
Xk_val_fft = list(np.fft.fft(x_N))

# print(Xk_values)
# print(Xk_val_fft)


frequences = np.linspace(0, fs, N)
continuous_amplitude = [np.abs(fourier_transform(freq)) for freq in frequences]
discrete_amplitude = [np.abs(discrete_transform(x_N, k)) for k in range(N)]


fig2, axs = plt.subplots(1, 2, tight_layout=True, figsize=(15, 7))
fig2.suptitle('Amplitude spectrum')

axs[0].grid()
axs[0].bar(frequences, continuous_amplitude, width=0.05)
axs[0].set_title('Continuous signal')
axs[0].set_xlabel('Frequency')
axs[0].set_ylabel('Amplitude')
axs[0].set_xlim([-1, fs])

axs[1].grid()
axs[1].bar(frequences, [abs(Xk) for Xk in Xk_values], width=0.05)
axs[1].set_title('Discrete signal')
axs[1].set_xlabel('Frequency')
axs[0].set_xlim([0, fs])

plt.show()

# Амплітудні спектри двох сигналів мають схожий вигляд в діапазоні [0, 3].
# Потім амплітуди дискретного сигналу починають знову зростати, у той час як
# амплітуди неперервної функції прямують до нуля і потім дорівнюють йому при
# всіх подальших значеннях частоти.
# Значення амплітуд значно різняться без додаткової нормалізації у вигляді 1/N


# часовий спектр із частотного
t_spectrum = list(np.fft.ifft(Xk_values))
# t_spectrum_fft = list(np.fft.ifft(Xk_val_fft))
print(t_spectrum)
print(x_N)

fig3, ax3 = plt.subplots(1, 2, figsize=(10, 6))

ax3[0].grid()
ax3[0].plot(np.arange(0, 10, step), t_spectrum)
ax3[0].set_title("x'[N]")
ax3[0].set_xlabel('t')
ax3[0].set_ylabel('x(t)')

ax3[1].grid()
ax3[1].plot(np.arange(0, 10, step), x_N)
ax3[1].set_title('x[N]')
ax3[1].set_xlabel('t')
ax3[1].set_ylabel('x(t)')

plt.show()

# форма графіків повністю співпадає







