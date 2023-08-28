import crocoddyl
import numpy as np


def logbarrier_test(nr):
    weights = np.ones(nr)
    bounds = 1.1 * np.ones(nr)
    damping = 0.5
    act_model = crocoddyl.ActivationModelLogBarrier(weights, bounds, damping)
    test_input = np.random.rand(nr)
    data = act_model.createData()
    # run the actual calculations
    act_model.calc(data, test_input)
    act_model.calcDiff(data, test_input)
    # print the results

    print("Dimension: {}".format(nr))
    print(data.a_value)
    print(data.Ar)
    print(data.Arr)


if __name__ == "__main__":
    # logbarrier_test(2)
    for dim in range(0, 10):
        logbarrier_test(dim)
