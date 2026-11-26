"""Compare clock jitter with python -m ImplicitRDP.scripts.benchmark_precise_sleep."""

import time
import numpy as np
from ImplicitRDP.common.precise_sleep import precise_sleep, precise_wait


def benchmark_sleep():
    dt = 0.1
    tol = 1e-3
    time_samples = list()
    for _ in range(100):
        precise_sleep(dt)
        time_samples.append(time.monotonic())
    time_deltas = np.diff(time_samples)

    from matplotlib import pyplot as plt
    plt.plot(time_deltas)
    plt.ylim((dt-tol,dt+tol))


def benchmark_wait():
    dt = 0.1
    tol = 1e-3
    errors = list()
    t_start = time.monotonic()
    for i in range(1,100):
        t_end_desired = t_start + i * dt
        time.sleep(max(0, t_end_desired - time.monotonic()))
        t_end = time.monotonic()
        errors.append(t_end - t_end_desired)

    new_errors = list()
    t_start = time.monotonic()
    for i in range(1,100):
        t_end_desired = t_start + i * dt
        precise_wait(t_end_desired)
        t_end = time.monotonic()
        new_errors.append(t_end - t_end_desired)

    from matplotlib import pyplot as plt
    plt.plot(errors, label='time.sleep')
    plt.plot(new_errors, label='sleep/spin hybrid')
    plt.ylim((-tol,+tol))
    plt.title('0.1 sec sleep error')
    plt.legend()


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    plt.figure()
    benchmark_sleep()
    plt.title('Intervals between precise_sleep calls')
    plt.figure()
    benchmark_wait()
    plt.show()
