import matplotlib.pyplot as plt
import numpy as np

from utils.solver.MPC.mpc_helper import LowPassFilter

if __name__ == "__main__":

    streaming_freq = 120.0
    # sine_freq = 2
    sine_freq = 1
    n_steps = int((1 / sine_freq) * streaming_freq)

    ts = np.linspace(0, 1 / sine_freq, n_steps)
    dt = ts[1] - ts[0]
    ys_perfect = np.cos(np.linspace(0, 2 * np.pi, n_steps))
    ys_noise = ys_perfect + np.random.uniform(-1e-3, 1e-3, size=(n_steps,))

    best_freq = 0
    lowest_err = np.inf

    vels = np.zeros(ys_noise.shape[0] - 1)
    fd_vels = np.zeros(ys_noise.shape[0] - 1)
    fd_vels_perfect = np.zeros(ys_noise.shape[0] - 1)
    # filter = LowPassFilter(np.ones(1) * 360) # 120
    for filter_freq in range(10, 360, 5):
        filter = LowPassFilter(np.ones(1) * filter_freq)
        filter.set_coefficients(1.0 / streaming_freq)

        for i in range(1, ys_noise.shape[0]):
            fd_vel_raw = (ys_noise[i] - ys_noise[i - 1]) / dt

            fd_vel = (ys_perfect[i] - ys_perfect[i - 1]) / dt

            vel_filterd = filter.compute_diff(fd_vel_raw)

            vels[i - 1] = vel_filterd
            fd_vels[i - 1] = fd_vel_raw
            fd_vels_perfect[i - 1] = fd_vel
        # plt.plot(ts[1:], fd_vels, label="fd_raw")
        # plt.plot(ts[1:], vels, label="fd filtered")
        # plt.plot(ts[1:], fd_vels_perfect, label="fd perfect")
        # plt.legend()
        # plt.show()
        err = np.linalg.norm(vels - fd_vels_perfect)
        print("Freq: {}, error: {}".format(filter_freq, err))
        if err < lowest_err:
            best_freq = filter_freq
    print(filter_freq)
