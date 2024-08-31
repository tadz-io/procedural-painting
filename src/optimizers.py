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
        initial_temp=1000,
        cooling_rate=0.95,
        rand_criterion=False,
        stop_criterion=1e-3,
    ) -> None:
        self._max_iterations = max_iterations
        self._loss = loss
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.rand_criterion: bool = rand_criterion
        self.stop_criterion = stop_criterion
        self._errors: list | None = None

    def loss(self, y_hat, y):
        return self._loss(y_hat, y)

    def optimize(self, canvas: Canvas, reference):
        # create list of errors at each iteration
        self._errors = []
        for i in range(self._max_iterations):
            # TODO: implement a function that takes a step in the current neighbourhood
            #      do not just assign random x, y coordinates
            x = np.random.randint(0, canvas.resolution[1])
            y = np.random.randint(0, canvas.resolution[0])
            canvas.draw.circle(center=(x, y), radius=5)
            # calculate current loss
            current_loss = self.loss(canvas._cache, reference)
            # calculate loss after drawing
            new_loss = self.loss(canvas.canvas, reference)
            # determine if radion acceptance criterion should be used
            if self.rand_criterion:
                acceptance_criterion = np.exp((current_loss - new_loss) / self.current_temp)
                print(f"acceptance criterion: {acceptance_criterion}\n")
            # TODO: remove if/else and set:
            #  acceptance_criterion = np.exp((current_loss - new_loss) / self.current_temp)
            else:
                # in this case not
                acceptance_criterion = 1e-9
            # evaluate the new loss against the current loss
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
