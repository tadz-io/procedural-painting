import cv2 as cv

from draw import Canvas
from optimizers import SimulatedAnnealing, mse

# maximum number of iterations
N_INTERATIONS = 5000

ref_image = cv.imread("./benchmark/img/example_001.png")


def benchmark_sa_optimizer():
    canvas = Canvas(cache_size=N_INTERATIONS)
    sa_optimizer = SimulatedAnnealing(
        max_iterations=N_INTERATIONS,
        loss=mse,
    )
    # do the optimization
    sa_optimizer.optimize(canvas=canvas, reference=ref_image)

    return
