import numpy as np


def test_fd_error(n):
    x = np.linspace(0, 2 * np.pi, num=n)
    y = np.sin(x)

    dxdy = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    dxdy_analytic = np.cos(np.linspace(0, 2 * np.pi, num=n))
    diff = np.abs(dxdy - dxdy_analytic[1:])
    print("Max error: %.3e" % np.max(diff))

    ddx = (dxdy[1:] - dxdy[:-1]) / (x[2:] - x[1:-1])
    ddx_analytic = -np.sin(np.linspace(0, 2 * np.pi, num=n))
    diff = np.abs(ddx - ddx_analytic[2:])
    print("Max error: %.3e" % np.max(diff))


if __name__ == "__main__":
    test_fd_error(1000)
    test_fd_error(10000)
    test_fd_error(100000)
