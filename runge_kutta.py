import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

C_symb = sp.symbols('C')
x0 = 1
xn = 2
y0 = 1.1
h = 0.01

def y(x, C):
    return x * sp.exp(x*C)


def f(x, y):
    return y/x * (sp.log(y/x) + 1)



def runge_kutta_method_2(x0, xn, y0, h):
    y_values = [y0]
    x_values = np.arange(x0, xn+h, h)
    for xi in x_values[1:]:
        yi = y_values[-1]
        yi_next = yi + h*f(xi+h/2, yi+h/2*f(xi, yi))
        y_values.append(yi_next)
    return x_values, y_values

def runge_kutta_method_4(x0, xn, y0, h):
    y_values = [y0]
    x_values = np.arange(x0, xn+h, h)
    for xi in x_values[1:]:
        yi = y_values[-1]
        k1 = h*f(xi, yi)
        k2 = h*f(xi+h/2, yi + k1/2)
        k3 = h*f(xi+h/2, yi+k2/2)
        k4 = h*f(xi+h, yi+k3)
        yi_next = yi + 1/6 * (k1 + 2*k2 + 2*k3 + k4)
        y_values.append(yi_next)
    return x_values, y_values


# check analytical solution
x_symb = sp.symbols('x')
y_symb = x_symb * sp.exp(C_symb*x_symb)
y_symb_diff = sp.diff(y_symb, x_symb)    # differentiate y_symb

f_symb = sp.simplify(y_symb/x_symb * (sp.log(y_symb/x_symb) + 1))  #substitute y in given diff_equation
print('Given diff.equation: ', f_symb)              # (log(exp(C*x)) + 1)*exp(C*x),     if simplify ln results are same
print('Diff.analytical solution: ', y_symb_diff)    # C*x*exp(C*x) + exp(C*x)
print()

# find C when x0, y0
C_value = (sp.solve(x0*sp.exp(C_symb*x0) - y0, C_symb))[0]
print('C = ', C_value)           # 0.0953101798043249
print()


# table
def result_table(function):
    x_values, y_values = function
    for xi, yi in zip(x_values, y_values):
        print('xi = ', xi, ' yi = ', yi, 'yi_an = ', y(xi, C_value))

# result_table(runge_kutta_method_2(x0, xn, y0, h))
# result_table(runge_kutta_method_4(x0, xn, y0, h))


# error
def R(x, h, order):
    func = sp.diff(x_symb * sp.exp(x_symb*C_value), x_symb, order+1) * (h**order) / sp.factorial(order)
    return func.subs(x_symb, x)

yn_values = [2.1, 2.245, 2.328, 2.382]
real_errors = [abs(2.42 - yn) for yn in yn_values]
h_values = [0.1, 0.05, 0.025, 0.01]

theoretical_error_2 = []
for h in h_values:
    theoretical_error_2.append(abs(R(xn, h, 2)))

theoretical_error_4 = []
for h in h_values:
    theoretical_error_4.append(abs(R(xn, h, 4)))


print()
print(h_values)
print('Real error: ', real_errors)
print('Theoretical error (2): ', theoretical_error_2)
print('Theoretical error (4): ', theoretical_error_4)

# graph 1
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].grid()
axs[0].plot(h_values, real_errors)
axs[0].set_xlabel('h')
axs[0].set_ylabel('Error')
axs[0].set_xticks(h_values)
axs[0].set_title('Real error dependency on h values')

axs[1].grid()
axs[1].plot(h_values, theoretical_error_2)
axs[1].set_xlabel('h')
axs[1].set_ylabel('Error')
axs[1].set_xticks(h_values)
axs[1].set_title('Theoretical (2) error dependency on h values')

plt.tight_layout()
plt.show()

# graph 2
fig2, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].grid()
ax[0].plot(h_values, real_errors)
ax[0].set_xlabel('h')
ax[0].set_ylabel('Error')
ax[0].set_xticks(h_values)
ax[0].set_title('Real error dependency on h values')

ax[1].grid()
ax[1].plot(h_values, theoretical_error_4)
ax[1].set_xlabel('h')
ax[1].set_ylabel('Error')
ax[1].set_xticks(h_values)
ax[1].set_title('Theoretical (4) error dependency on h values')

plt.tight_layout()
plt.show()







