import numpy as np
from typing import Callable
from src.draw import Canvas


def mse(x, y):
    x_uint16 = x.astype(np.uint16)
    y_uint16 = y.astype(np.uint16)
    return np.mean((x_uint16 - y_uint16) ** 2)


class SimulatedAnnealing:
    def __init__(
        self,
        max_iterations: int,
        loss: Callable,
        initial_temp: int = 1000,
        cooling_rate: float = 0.999,
        sampler: Callable[..., int] = lambda: (np.random.binomial(100, 0.5, 1) - 50)[0],
        stop_criterion: float = 1e-3,
    ) -> None:
        self._max_iterations = max_iterations
        self._loss = loss
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.sampler = sampler
        self.stop_criterion = stop_criterion
        self._errors: list = []
        self._entropy_norm = []

    def loss(self, y_hat, y):
        return self._loss(y_hat, y)

    @property
    def error_stats(self):
        if not self._errors:
            return 0.0, 0.0
        return np.mean(self._errors), np.std(self._errors)

    @error_stats.setter
    def error_stats(self, loss: float) -> None:
        self._errors.append(loss)

    def optimize(self, canvas: Canvas, reference):
        # create list of errors at each iteration
        self._errors = []
        x = np.random.randint(0, canvas.resolution[1])
        y = np.random.randint(0, canvas.resolution[0])
        color = np.random.randint(0, 256)
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
            color_old = color
            # take a random step along the color gradient
            color = np.clip(color + np.random.randint(-5, 5, 1), 0, 255)[0]
            color_tuple = (int(color),) * 3
            canvas.draw.circle(center=(x, y), color=color_tuple, radius=5)
            # calculate previous loss
            previous_loss = self.loss(canvas.previous, reference)
            # update list with previous loss value
            self.error_stats = previous_loss
            # get standard deviation of loss
            _, loss_std = self.error_stats
            # calculate loss after drawing
            new_loss = self.loss(canvas.canvas, reference)
            # calculate delta in loss and normalize
            d_e = new_loss - previous_loss  # / (loss_std + 1e-8)
            # Evaluate the new loss against the previous loss
            if new_loss > previous_loss:
                # calculate new acceptance criteria
                acceptance_criterion = np.exp(-(d_e) / self.current_temp)
                if np.random.rand() < acceptance_criterion:
                    # Case 1: Accept new x and y but revert the canvas
                    canvas.revert()
                    # canvas.draw.circle(center=(x, y), radius=5, color=(0, 0, 255))
                    # Keep x and y as they are (x_new, y_new)
                else:
                    # Case 2: Revert both the canvas and the coordinates
                    canvas.revert()
                    # canvas.draw.circle(center=(x, y), radius=5, color=(255, 0, 0))
                    # Revert x and y to the values before the random step
                    x, y = x_old, y_old
                    color = color_old
                # add new entropy criterion to list
                self._entropy_norm.append(acceptance_criterion)
            # cool down temperature
            self.current_temp *= self.cooling_rate

            # stop optimization of threshold is reached
            if new_loss < self.stop_criterion:
                print(f"Stopping early at iteration {i} with error {new_loss}")
                break

        return canvas
