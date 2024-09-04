import cv2 as cv  # noqa
import numpy as np
from numpy.typing import NDArray
from typing import Callable

# load reference image to optimize against
# reference_image = Image.open("src/img/example_001.png").convert("RGB")


class Canvas:
    def __init__(self, resolution=(600, 800), bg_color=(255, 255, 255), cache_size=5000) -> None:
        self.resolution = resolution
        self.bg_color = bg_color
        self._cache_size = cache_size
        self._canvas: NDArray | None = None
        # setup blank canvas
        self._state_history: NDArray = np.empty((cache_size, *resolution, 3), dtype=np.uint8)
        self._history_index = 0  # track current position in history
        self._history_count = 0  # track number of stored states
        # instatiate Drawer class
        self.draw = Drawer(self)
        self.clear()

    @property
    def canvas(self):
        return self._canvas

    @property
    def history(self):
        return self._state_history[: self._history_count, ...]

    @property
    def previous_state(self):
        if self._history_count > 0:
            return self._state_history[self._history_index - 1]
        else:
            print("no previous state")

    def clear(self):
        """Create a blank canvas with the specified background color"""
        self._canvas = np.ones(self.resolution + (3,), np.uint8) * np.array(self.bg_color, np.uint8)
        self._save_current_state()

    def revert(self):
        """revert canvas to the cache state"""
        if self._history_count > 0:
            self._history_index = (self._history_index - 1) % self._cache_size
            self._history_count -= 1
            self._canvas = self._state_history[self._history_index].copy()
        else:
            print("no previous state to revert to")

    def _save_current_state(self):
        """Save the current state of the canvas in history"""
        self._state_history[self._history_index] = self._canvas.copy()
        # increaste counter
        self._history_index = (self._history_index + 1) % self._cache_size
        if self._history_count < self._cache_size:
            self._history_count += 1


class Drawer:
    def __init__(self, parent: Canvas) -> None:
        self.parent: Canvas = parent

    def _cache_canvas_state(method: Callable) -> Callable:
        """decorator to cache the state of the canvas before drawing"""

        def wrapper(self, *args, **kwargs):
            # save curreng state of canvas to _cache before drawing
            result = method(self, *args, **kwargs)
            self.parent._save_current_state()

            return result

        return wrapper

    @_cache_canvas_state
    def line(self, pt1, pt2, color=(255, 255, 255), thickness=-1):
        # Draw a line on the parent canvas
        cv.line(self.parent._canvas, pt1, pt2, color, thickness)

    @_cache_canvas_state
    def rectangle(self, pt1, pt2, color=(0, 0, 0), thickness=-1):
        # Draw a rectangle on the parent canvas
        cv.rectangle(self.parent._canvas, pt1, pt2, color, thickness)

    @_cache_canvas_state
    def circle(self, center, radius, color=(0, 0, 0), thickness=-1):
        # Draw a circle on the parent canvas
        cv.circle(
            self.parent._canvas,
            center,
            radius,
            color,
            thickness,
        )
