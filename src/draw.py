import cv2 as cv  # noqa
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Any, Optional


class Canvas:
    def __init__(self, resolution=(600, 800), bg_color=(255, 255, 255), cache_size=5000) -> None:
        self.resolution = resolution
        self.bg_color = bg_color
        self._cache_size = cache_size
        self._canvas: NDArray = np.ones(self.resolution + (3,), dtype=np.uint8) * np.array(
            self.bg_color, np.uint8
        )
        # Setup blank canvas
        self._state_history: NDArray = np.empty((cache_size, *resolution, 3), dtype=np.uint8)
        self._history_index = 0  # Tracks the current state in the history
        self._cache_current_state()  # Save the initial state

        # Instantiate Drawer class
        self.draw = Drawer(self)

    @property
    def canvas(self) -> NDArray:
        """Returns the latest canvas state."""
        return self._canvas

    @property
    def previous(self) -> Optional[NDArray]:
        """Returns the previous canvas state in history if available."""
        if self._history_index > 1:
            return self._state_history[self._history_index - 2]
        return None

    def clear(self) -> None:
        """Clears the canvas to the background color."""
        self._canvas[:] = np.array(self.bg_color, dtype=np.uint8)
        self._cache_current_state()

    def revert(self) -> None:
        """Reverts the canvas to the previous cached state."""
        if self._history_index > 1:
            self._history_index -= 1
            self._canvas[:] = self._state_history[self._history_index - 1]

    def _cache_current_state(self) -> None:
        """Saves the current state of the canvas in history."""
        # Ensure index is within bounds and maintain the correct behavior
        if self._history_index < self._cache_size:
            self._state_history[self._history_index] = self._canvas.copy()
            self._history_index += 1
        else:
            # Shift the history to accommodate new states if cache size exceeded
            self._state_history[:-1] = self._state_history[1:]
            self._state_history[-1] = self._canvas.copy()

    def save(self, filename: str) -> None:
        """Save the current canvas state to an image file."""
        cv.imwrite(filename, self._canvas)


class Drawer:
    def __init__(self, parent: Canvas) -> None:
        self.parent: Canvas = parent

    @staticmethod
    def _cache_canvas_state(method: Callable) -> Callable:
        """Decorator to save the current state of the canvas after drawing operations."""

        def wrapper(self, *args, **kwargs) -> Any:
            result = method(self, *args, **kwargs)  # Call the original drawing method
            self.parent._cache_current_state()  # Save the current state of the canvas
            return result

        return wrapper

    @_cache_canvas_state
    def line(self, pt1, pt2, color=(255, 255, 255), thickness=-1) -> None:
        """Draws a line on the parent canvas."""
        cv.line(self.parent._canvas, pt1, pt2, color, thickness)

    @_cache_canvas_state
    def rectangle(self, pt1, pt2, color=(0, 0, 0), thickness=-1) -> None:
        """Draws a rectangle on the parent canvas."""
        cv.rectangle(self.parent._canvas, pt1, pt2, color, thickness)

    @_cache_canvas_state
    def circle(self, center, radius, color=(0, 0, 0), thickness=-1) -> None:
        """Draws a circle on the parent canvas."""
        cv.circle(self.parent._canvas, center, radius, color, thickness)
