import numpy as np
from typing import Callable
from src.draw import Canvas


def mse(x, y):
    return np.mean(np.sum((x - y) ** 2))


class SimulatedAnnealing:
    def __init__(
        self,
        max_iterations: int,
        loss: Callable,
        initial_temp: int = 1000,
        cooling_rate: float = 0.95,
        sampler: Callable[..., int] = lambda: (np.random.binomial(100, 0.5, 1) - 50)[0],
        stop_criterion: float = 1e-3,
    ) -> None:
        self._max_iterations = max_iterations
        self._loss = loss
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.sampler = sampler
        self.stop_criterion = stop_criterion
        self._errors: list | None = None

    def loss(self, y_hat, y):
        return self._loss(y_hat, y)

    def optimize(self, canvas: Canvas, reference):
        # create list of errors at each iteration
        self._errors = []
        x = np.random.randint(0, canvas.resolution[1])
        y = np.random.randint(0, canvas.resolution[0])
        # start iterating
        for i in range(self._max_iterations):
            # TODO: implement a function that takes a step in the current neighbourhood
            #      do not just assign random x, y coordinates
            # add random step to coordinates, only if it does not exceed canvas dimensions
            while True:
                x_new = x + self.sampler()
                y_new = y + self.sampler()
                # check if canvas bounds are not exceeded by random steps
                if (0 <= x_new <= canvas.resolution[1]) and (0 <= y_new <= canvas.resolution[0]):
                    x, y = x_new, y_new
                    break
            canvas.draw.circle(center=(x, y), radius=5)
            # calculate current loss
            current_loss = self.loss(canvas._cache, reference)
            # calculate loss after drawing
            new_loss = self.loss(canvas.canvas, reference)
            # calculate new acceptance criteria
            acceptance_criterion = np.exp((current_loss - new_loss) / self.current_temp)
            # evaluate the new loss against the current lossş
            if new_loss < current_loss or np.random.rand() < acceptance_criterion:
                # accept the new state, we don't have to revert the state of the canvas
                self._errors.append(new_loss)
            else:
                # revert canvas to previous state
                canvas.revert()
            # cool down temperature
            self.current_temp *= self.cooling_rate

            # stop optimization of threshold is reached
            if new_loss < self.stop_criterion:
                print(f"Stopping early at iteration {i} with error {new_loss}")
                break

        return canvas
