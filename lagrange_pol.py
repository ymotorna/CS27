import math
import numpy as np
import matplotlib.pyplot as plt

def lagrange_pol(x_val, y_val, x):
    L = 0
    for xi, yi in zip(x_val, y_val):
        l = 1
        for a in x_val:
            if a != xi:
                l = l * (x - a) / (xi - a)
        L += yi * l
    return L

def linear_interpolation(x_val, y_val, x):
    for i in range(len(x_val)):
        if x >= x_val[i] and x <= x_val[i+1]:
            xi = x_val[i]
            xi1 = x_val[i+1]
            return y_val[i] * (x-xi1)/(xi-xi1) + y_val[i+1] * (x-xi)/(xi1-xi)



# f(x) = x^2 * e^(-x^2), x є [a; b]
def f(x):
    return x**2 * math.exp(-x**2)


def nodes(x0, xn, n):      #calculate nodes
    node_list = [x0]
    h = (xn - x0) / n
    for i in range(1, n):
        xi = x0 + i*h
        node_list.append(xi)

    node_list.append(xn)
    return node_list


# point out max values
def max_points(x_list, y_list, max_value, graph, ax, point_name):
    for x, y in zip(x_list, y_list):
        if y == max_value:
            ax[graph].scatter(x, y, label=point_name+' max', s=30, color='red')
        ax[graph].legend()



# set variables
a = -3
b = 3
n = 5

x_values = list(np.arange(a, b+0.01, 0.01))
y_values = [f(x) for x in x_values]

x_nodes = nodes(a, b, n)
y_nodes = [f(x) for x in x_nodes]

L_values = [lagrange_pol(x_nodes, y_nodes, x) for x in x_values]
errors = [abs(y_values[i] - L_values[i]) for i in range(len(x_values))]
max_error = max(errors)

Ll_values = [linear_interpolation(x_nodes, y_nodes, x) for x in x_values]
errors_l = [abs(y_values[i] - Ll_values[i]) for i in range(len(x_values))]
max_error_l = max(errors_l)


# plot graph
fig, axd = plt.subplot_mosaic(
    '''
    AB
    CD
    ''',
    figsize=(12, 8)
)

#graph 1
axd['A'].grid()
axd['A'].plot(x_values, y_values, label='f(x)')
axd['A'].plot(x_values, L_values, label='L(x)')
axd['A'].set_xticks(range(a, b+1))
axd['A'].set_xlabel('x', weight='bold')
axd['A'].set_ylabel('y', weight='bold')
axd['A'].legend()

#graph 2
axd['B'].grid()
axd['B'].plot(x_values, errors, label='| f(x)-L(x) |')
max_points(x_values, errors, max_error, 'B', axd, 'Error')
axd['B'].set_xlabel('x', weight='bold')
axd['B'].set_ylabel('Error', weight='bold')
axd['B'].legend()

fig.text(0.52, 0.96, 'Global Interpolation', ha='center', va='center', fontsize=15, weight='bold')

#graph 3
axd['C'].grid()
axd['C'].plot(x_values, y_values, label='f(x)')
axd['C'].plot(x_values, Ll_values, label='L(x)')
axd['C'].set_xlabel('x', weight='bold')
axd['C'].set_ylabel('y', weight='bold')
axd['C'].legend()

#graph 4
axd['D'].grid()
axd['D'].plot(x_values, errors_l, label='| f(x)-L(x) |')
max_points(x_values, errors_l, max_error_l, 'D', axd, 'Error')

axd['D'].set_xlabel('x', weight='bold')
axd['D'].set_ylabel('Error', weight='bold')
axd['D'].legend()


fig.text(0.5, 0.48, 'Piecewise Linear Interpolation',  ha='center', va='center', fontsize=15, weight='bold')


plt.subplots_adjust(hspace=23)
plt.tight_layout(rect=[0, 0.02, 1, 0.98])
plt.show()


print('ERROR_max: ', max_error)
print('Linear ERROR_max: ', max_error_l)


#extrapolation
ae = 2*a-b
be = 2*b-a

x_extended = list(np.arange(ae, be+0.01, 0.01))
y_extended = [f(x) for x in x_extended]
L_extrapolation = [lagrange_pol(x_nodes, y_nodes, x) for x in x_extended]
x_ex_left = [x for x in x_extended if x <= a]
x_ex_right = [x for x in x_extended if x >= b]
errors_ex_left = [abs(y_extended[i] - L_extrapolation[i]) for i in range(len(x_extended)) if x_extended[i] < a]
errors_ex_right = [abs(y_extended[i] - L_extrapolation[i]) for i in range(len(x_extended)) if x_extended[i] > b]
error_max_right = max(errors_ex_right)
error_max_left = max(errors_ex_left)

fig2, axs = plt.subplot_mosaic(
    '''
    ABC
    ''',
    figsize=(15, 5)
)

#graph 1
axs['A'].grid()
axs['A'].plot(x_extended, y_extended, label='f(x)')
axs['A'].plot(x_extended, L_extrapolation, label='L(x)')
axs['A'].set_xticks(range(ae, be))
axs['A'].set_xlabel('x', weight='bold')
axs['A'].set_ylabel('y', weight='bold')

axs['A'].legend()


#graph 2
axs['B'].grid()
axs['B'].plot(x_ex_left, errors_ex_left, label='| f(x)-L(x) |')
max_points(x_ex_left, errors_ex_left, error_max_left, 'B', axs, 'Error')
axs['B'].set_xlabel('x', weight='bold')
axs['B'].set_ylabel('Error', weight='bold')
axs['B'].legend()
axs['B'].set_title('Extrapolation', weight='bold', fontsize=15)

#graph 3
axs['C'].grid()
axs['C'].plot(x_ex_right, errors_ex_right, label='| f(x)-L(x) |')
max_points(x_ex_right, errors_ex_right, error_max_right, 'C', axs, 'Error')
axs['C'].set_xlabel('x', weight='bold')
axs['C'].set_ylabel('Error', weight='bold')
axs['C'].legend()


# fig.text(0.5, 1.05, 'Extrapolation',  ha='center', va='bottom', fontsize=16, weight='bold')


plt.tight_layout()
plt.show()


print('ERROR_max_left: ', error_max_left)
print('ERROR_max_right: ', error_max_right)





