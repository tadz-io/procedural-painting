import numpy as np
from numpy.typing import NDArray
from typing import Callable
from draw import Canvas
from typing import Dict, Tuple


def mse(x: NDArray, y: NDArray):
    # need to convert to 16 bit uints when taking square of 8 bit arrays
    x_uint16 = x.astype(np.uint16)
    y_uint16 = y.astype(np.uint16)
    return np.mean((x_uint16 - y_uint16) ** 2)


class Parameter:
    def __init__(
        self,
        initial: int | Tuple[int, ...],
        step_size: int | Tuple[int, ...],
        min_value: int | Tuple[int, ...],
        max_value: int | Tuple[int, ...],
    ) -> None:
        """
        Parameter class to hold the initial value and step size for optimization.

        Args:
            initial: Initial value of the parameter.
            step_size: Step size for the parameter. If 0, parameter doesn't change.
        """
        self.value = initial
        self.step_size = step_size
        self.min_value = min_value
        self.max_value = max_value

    def update(self, sampler: Callable[[int, int], int]) -> None:
        """
        Generate a random step within the step_size range using the sampler method.

        Args:
            sampler: The callable to generate random values.
        """
        # If the value is an int, handle as a single parameter
        if isinstance(self.value, int):
            if self.step_size != 0:
                step = sampler(-self.step_size, self.step_size)
                self.value = int(np.clip(self.value + step, self.min_value, self.max_value))

        # If the value is a tuple, handle as multiple parameters
        elif isinstance(self.value, tuple):
            # TODO: check if stepsize is 0, if so do not update
            # If step_size is an int, apply the same step to all elements in the tuple
            if isinstance(self.step_size, int):
                step = sampler(-self.step_size, self.step_size)
                self.value = tuple(
                    int(np.clip(v + step, min_v, max_v))
                    for v, min_v, max_v in zip(self.value, self.min_value, self.max_value)
                )

            # If step_size is a tuple, generate a step for each element
            elif isinstance(self.step_size, tuple):
                # TODO: check if stepsize is 0, if so do not update
                self.value = tuple(
                    int(np.clip(v + sampler(-s, s), min_v, max_v))
                    for v, s, min_v, max_v in zip(
                        self.value, self.step_size, self.min_value, self.max_value
                    )
                )


class SimulatedAnnealing:
    def __init__(
        self,
        max_iterations: int,
        loss: Callable,
        initial_temp: int = 1000,
        cooling_rate: float = 0.999,
        sampler: Callable[[int, int], int] = np.random.randint,
        stop_criterion: float = 1e-3,
    ) -> None:
        self._max_iterations = max_iterations
        self._loss = loss
        self.current_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.sampler = sampler
        self.stop_criterion = stop_criterion
        self._errors: list = []

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

    def optimize(
        self, canvas: Canvas, reference: NDArray, parameters: Dict[str, Parameter], brush: str
    ):
        # create list of errors at each iteration
        self._errors = []

        # start iterating
        for i in range(self._max_iterations):
            # store parameter values before updating
            params_previous = {k: param.value for k, param in parameters.items()}
            # update parameters by taking a random step
            for param in parameters.values():
                param.update(self.sampler)
            # get the drawing method
            drawer = getattr(canvas.draw, brush)
            kwargs = {k: param.value for k, param in parameters.items()}
            # call the draw method with the kwargs
            drawer(**kwargs)
            # calculate previous loss
            previous_loss = self.loss(canvas.previous, reference)
            # update list with previous loss value
            self.error_stats = previous_loss
            # calculate loss after drawing
            new_loss = self.loss(canvas.canvas, reference)
            # calculate delta in loss
            d_e = new_loss - previous_loss
            # Evaluate the new loss against the previous loss
            if new_loss > previous_loss:
                # calculate new acceptance criteria
                acceptance_criterion = np.exp(-d_e / self.current_temp)
                if np.random.rand() < acceptance_criterion:
                    # Case 1: Accept new parameters but revert the canvas
                    canvas.revert()
                else:
                    # Case 2: Revert both the canvas and the parameters
                    canvas.revert()
                    for k, param in parameters.items():
                        param.value = params_previous[k]
            # cool down temperature
            self.current_temp *= self.cooling_rate

            # stop optimization if threshold is reached
            if new_loss < self.stop_criterion:
                print(f"Stopping early at iteration {i} with error {new_loss}")
                break

        return canvas
