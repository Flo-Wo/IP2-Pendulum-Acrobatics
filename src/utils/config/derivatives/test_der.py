import numpy as np


frequencies = np.array([4, 4, 4, 4])
n = 2000

t_support = np.linspace(0, 2 * np.pi, num=n)
trajectory = np.sin(frequencies[None, :] * t_support[:, None])

dt = np.diff(t_support, append=t_support[:1])[:, None]
dt_const = 2 * np.pi / n
print("dt")
print(dt_const)

diff = np.diff(trajectory, axis=0, append=trajectory[1:2, :])
first_der = np.diff(trajectory, axis=0, append=trajectory[1:2, :]) / dt_const

second_der = np.diff(first_der, axis=0, append=first_der[1:2, :]) / dt_const

first_der_an = frequencies * np.cos(frequencies[None, :] * t_support[:, None])
second_der_an = (
    frequencies**2 * (-1) * np.sin(frequencies[None, :] * t_support[:, None])
)

print("diff: ", np.linalg.norm(first_der[:, 0] - first_der_an[:, 0], ord=np.inf))
print("diff: ", np.linalg.norm(second_der[:, 0] - second_der_an[:, 0], ord=np.inf))
