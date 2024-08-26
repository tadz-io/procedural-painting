from PIL import Image, ImageDraw
import numpy as np


class Canvas:
    def __init__(self, n_strokes, resolution=(800, 600), bg_color=(255, 255, 255)) -> None:
        self.n_strokes = n_strokes
        self.resolution = resolution
        self.bg_color = bg_color
        self._canvas = None
        self._draw = None
        self.clear()

    @property
    def canvas(self):
        return self._canvas

    def clear(self):
        # draw = ImageDraw.Draw(image)
        self._canvas = Image.new("RGB", self.resolution, self.bg_color)
        self._draw = ImageDraw.Draw(self.canvas)

    def draw_points(self, radius=5):
        """Draw random points (circles) on the canvas with a specified radius."""
        for _ in range(100):
            x = np.random.randint(0, self.resolution[0])
            y = np.random.randint(0, self.resolution[1])

            # Calculate the bounding box for the circle
            upper_left = (x - radius, y - radius)
            lower_right = (x + radius, y + radius)

            # Draw the circle
            self._draw.ellipse([upper_left, lower_right], fill=(0, 0, 0))
