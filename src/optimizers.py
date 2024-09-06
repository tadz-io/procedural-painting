import numpy as np
from typing import Callable
from src.draw import Canvas


def mse(x, y):
    return np.mean((x - y) ** 2)


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
            while True:
                x_old, y_old = x, y
                x_new = x + self.sampler()
                y_new = y + self.sampler()
                # check if canvas bounds are not exceeded by random steps
                if (0 <= x_new <= canvas.resolution[1]) and (0 <= y_new <= canvas.resolution[0]):
                    x, y = x_new, y_new
                    break
            canvas.draw.circle(center=(x, y), radius=5)
            # calculate previous loss
            previous_loss = self.loss(canvas.previous, reference)
            # calculate loss after drawing
            new_loss = self.loss(canvas.canvas, reference)
            # calculate delta in loss
            d_e = (new_loss - previous_loss) / 1
            self._errors.append(d_e)
            # Evaluate the new loss against the previous loss
            if new_loss > previous_loss:
                # calculate new acceptance criteria
                acceptance_criterion = np.exp(-(d_e) / self.current_temp)
                if np.random.rand() < acceptance_criterion:
                    # Case 1: Accept new x and y but revert the canvas
                    canvas.revert()
                    canvas.draw.circle(center=(x, y), radius=5, color=(0, 0, 255))
                    # Keep x and y as they are (x_new, y_new)
                else:
                    # Case 2: Revert both the canvas and the coordinates
                    canvas.revert()
                    # Revert x and y to the values before the random step
                    x, y = x_old, y_old
            # cool down temperature
            self.current_temp *= self.cooling_rate

            # stop optimization of threshold is reached
            if new_loss < self.stop_criterion:
                print(f"Stopping early at iteration {i} with error {new_loss}")
                break

        return canvas
