import glm
from OpenGL.GL import *
import numpy as np

class ColorPickingManager:
    """
    Manages picking mode: draws each cube with a unique color ID, 
    reads back color and depth at cursor, identifies the clicked sub-cube.
    Also handles translating/rotating the picked cube under the mouse.
    """
    def __init__(self, rubiks_renderer):
        self.rubiks_renderer = rubiks_renderer
        self.enabled = False
        self.picked_cube_index = None
        self.picked_cube_depth = None

    def toggle(self):
        self.enabled = not self.enabled
        self.picked_cube_index = None
        self.picked_cube_depth = None
        print("Picking Mode:", self.enabled)

    def pick(self, window, x, y, width, height):
        """
        Render in picking mode, read the color/depth at (x, y).
        Convert color to sub-cube index.
        """
        # 1) Bind an offscreen framebuffer or just use the default 
        #    but be sure to clear it first.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # 2) Draw each sub-cube with a unique color = index2color(i).
        self.rubiks_renderer.draw_picking_pass()

        # 3) Read pixel color & depth
        # Note that OpenGL coordinates have (0,0) at bottom-left, while 
        # many window systems use top-left for the mouse coordinate system.
        # So we might need to flip the Y coordinate:
        flipped_y = int(height - y - 1)

        pixel_color = glReadPixels(
            x, flipped_y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE
        )
        pixel_depth = glReadPixels(
            x, flipped_y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT
        )
        if pixel_color is not None and pixel_depth is not None:
            color = tuple(pixel_color[0])
            depth = pixel_depth[0][0]
            # Convert color -> index
            i = self.color_to_index(color)
            self.picked_cube_index = i
            self.picked_cube_depth = depth
            print(f"Picked index={i}, color={color}, depth={depth}")
        else:
            self.picked_cube_index = None
            self.picked_cube_depth = None

    def index_to_color(self, i):
        """
        Encode index i (0..255^3-1) into an RGB color.
        We'll just store each channel in [0..255].
        """
        r = (i & 0x000000FF) >> 0
        g = (i & 0x0000FF00) >> 8
        b = (i & 0x00FF0000) >> 16
        return (r, g, b)

    def color_to_index(self, color):
        """
        Decode color to integer index.
        color is (r, g, b)
        """
        r, g, b = color
        return (b << 16) + (g << 8) + (r << 0)

    def translate_picked_cube(self, dx, dy, camera):
        """
        Translate the picked cube, so it "follows" the mouse under the camera.
        We'll do a simplified approach: treat the depth as a reference 
        and do an unproject from screen space. 
        In a real system, you'd do more robust handling with a plane intersection.
        """
        if self.picked_cube_index is None:
            return

        # Convert screen dx,dy to [ -1..1 ] normalized device coords
        ndx = dx / camera.width * 2.0
        ndy = -dy / camera.height * 2.0

        # We'll shift the cube in the XY directions in view space
        # proportionally to ndx, ndy, ignoring perspective for simplicity
        scube = self.rubiks_renderer.data.sub_cubes[self.picked_cube_index]
        # Tweak these factors if you want different sensitivity
        factor = 2.0
        scube.position.x += ndx * factor
        scube.position.y += ndy * factor
        scube.update_model_matrix()

    def rotate_picked_cube(self, dx, dy, camera):
        """
        Rotate the picked cube around its local X/Y (or global).
        We'll do a simple approach: left-right motion = rotate Y,
        up-down motion = rotate X.
        """
        if self.picked_cube_index is None:
            return

        scube = self.rubiks_renderer.data.sub_cubes[self.picked_cube_index]
        scube.rotation.y += dx * 0.2
        scube.rotation.x += dy * 0.2
        scube.update_model_matrix()
