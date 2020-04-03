import time
import numpy as np
from numpy.random import uniform

from acrobotics.shapes import Box
from acrolib.geometry import pose_x

box1 = Box(1, 0.5, 2)
box2 = Box(0.7, 1.5, 1)


def check_random_collision():
    tf1 = pose_x(uniform(-2, 2), uniform(0, 3), 0, 0)
    tf2 = pose_x(uniform(-2, 2), 0, uniform(0, 3), 0)

    return box1.is_in_collision(tf1, box2, tf2)


N = 1000
run_times = np.zeros(N)
results = np.zeros(N, dtype=bool)
for i in range(N):
    start = time.time()
    res = check_random_collision()
    end = time.time()

    run_times[i] = end - start
    results[i] = res

print(f"{N} runs took an average of {np.mean(run_times)*1000} ms per run")
print(f"Not in collion average: {1000 * np.mean(run_times[results == False])}")
print(f"In collision average: {1000 * np.mean(run_times[results == True])}")
