import glm
from OpenGL.GL import *
import numpy as np
import ctypes

class ColorPickingManager:
    """
    Manages color picking mode:
      - toggle picking
      - when picking is on, we do an offscreen pass with unique "ID" colors
      - read pixel color => convert to sub-cube index
      - store depth => used for translating the object under the mouse
      - rotate_picked_cube(...) or translate_picked_cube(...) if user drags
    """
    def __init__(self, rubiks_renderer):
        self.renderer = rubiks_renderer  # type: RubiksCubeRenderer
        self.enabled = False
        self.picked_id = None
        self.picked_depth = None

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.picked_id = None
            self.picked_depth = None
        print("Picking mode:", self.enabled)

    def pick(self, window, x, y, win_width, win_height):
        """
        Render a unique color for each sub-cube at the pixel under (x,y).
        Read that pixel color => decode sub-cube index => store it in self.picked_id.
        Also read the depth => store in self.picked_depth.
        """
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Use the same VAO, but set picking uniform
        glUseProgram(self.renderer.shader)
        loc_pick = glGetUniformLocation(self.renderer.shader, "u_PickingMode")
        glUniform1i(loc_pick, 1)  # turn on picking

        loc_pick_col = glGetUniformLocation(self.renderer.shader, "u_PickingColor")
        loc_mvp = glGetUniformLocation(self.renderer.shader, "u_MVP")
        loc_color = glGetUniformLocation(self.renderer.shader, "u_Color")
        glUniform4f(loc_color, 1, 1, 1, 1)  # not used in picking

        glBindVertexArray(self.renderer.vao)

        # We do NOT bind the usual texture, because we only want solid colors
        # for picking. The sampler is ignored if picking mode is on, but it's
        # safe to unbind the texture.

        # Draw each sub-cube with a unique picking color (R, G, B).
        for sub_cube in self.renderer.data.sub_cubes:
            # unique color
            idx = sub_cube.idx
            # Convert idx to an RGB color. 
            # E.g. 24-bit: R = idx & 255, G = (idx >> 8) & 255, B = (idx >> 16) & 255
            r = (idx & 0x000000FF) / 255.0
            g = ((idx & 0x0000FF00) >> 8) / 255.0
            b = ((idx & 0x00FF0000) >> 16) / 255.0

            glUniform3f(loc_pick_col, r, g, b)

            mvp = glm.mat4(1.0)
            view_proj = glm.mat4(1.0)
            # In main.py, we have the camera's view/proj. But here, we don't have direct access.
            # So we pass them as arguments, or re-render the scene with the same camera's transforms
            # The simpler approach is to call picking_manager.pick() from within the same frame 
            # you have the camera's view/proj. So let's do a small trick:
            # We can read the current states from the GL if you keep them around.
            # But the user code calls picking right after we have the camera. 
            # We'll just replicate the approach from the normal draw:

            # Instead, we can rely on the fact that main.py calls picking_manager.pick(...) 
            # inside the mouse_button_callback, and that the camera is not changed in between.
            # A hacky approach is: re-use the last known (view, proj).
            # For completeness, let's store them in the manager. 
            # But let's keep it simple: store them in self.cached_view, self.cached_proj 
            # when the scene is rendered. Then use them here:
            if hasattr(self, 'cached_view') and hasattr(self, 'cached_proj'):
                view_proj = self.cached_proj * self.cached_view  # (project * view) is the usual order
            else:
                # fallback to identity
                pass

            # MVP for that sub-cube
            sub_model = sub_cube.model_matrix
            mvp = (self.cached_proj) * (self.cached_view) * sub_model if \
                  hasattr(self, 'cached_view') else glm.mat4(1.0)

            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm.value_ptr(mvp))

            glDrawElements(GL_TRIANGLES,
                           len(self.renderer.indices),
                           GL_UNSIGNED_INT,
                           ctypes.c_void_p(0))

        # Read color & depth
        # Note that the GL's coordinates start at lower-left corner for (0,0).
        # If your window system is the same, fine. If not, invert y with height-y-1.
        y_flipped = int(win_height - y - 1)

        pixel_color = glReadPixels(x, y_flipped, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE)
        pixel_depth = glReadPixels(x, y_flipped, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)

        # Convert from bytes to an integer ID
        if pixel_color is not None:
            r = pixel_color[0]
            g = pixel_color[1]
            b = pixel_color[2]
            picked_idx = (r) | (g << 8) | (b << 16)
            self.picked_id = picked_idx
            print("Picked sub-cube index:", picked_idx)
        else:
            self.picked_id = None

        if pixel_depth is not None:
            self.picked_depth = pixel_depth[0]
            print("Picked depth:", self.picked_depth)
        else:
            self.picked_depth = None

        # Turn off picking mode again
        glUniform1i(loc_pick, 0)
        glBindVertexArray(0)

    def rotate_picked_cube(self, dx, dy, camera):
        """
        Called from cursor_position_callback while left mouse is pressed 
        if we have a picked cube.
        Rotates that cube around the camera view, for example.
        """
        if self.picked_id is None:
            return
        # Retrieve sub-cube
        sub_cube = self.renderer.data.get_sub_cube_by_index(self.picked_id)
        if not sub_cube:
            return

        # Make some rotation transform based on dx, dy 
        # For example, rotate around camera's right vector or up vector.
        # We'll do a simple approach: horizontal mouse => rotate around Y, vertical => rotate around X
        angle_x = dx * 0.2  # tune
        angle_y = dy * 0.2

        # Build the rotation around global axes for simplicity
        rot_x = glm.rotate(glm.mat4(1.0), glm.radians(angle_y), glm.vec3(1,0,0))
        rot_y = glm.rotate(glm.mat4(1.0), glm.radians(angle_x), glm.vec3(0,1,0))

        # Combine
        rotation_mat = rot_y * rot_x

        # Apply this rotation to the sub-cube visually only
        # A naive approach is: sub_cube.model_matrix = rotation_mat * sub_cube.model_matrix
        # But that will accumulate. 
        # If you want to truly keep track, you might store a base transform. For now, let's do naive:

        sub_cube.model_matrix = rotation_mat * sub_cube.model_matrix

    def translate_picked_cube(self, dx, dy, camera):
        """
        Called from cursor_position_callback while right mouse is pressed 
        if we have a picked cube.
        We want to drag the sub-cube in the plane of the screen, for example.
        This is more complex if we want the "cube to stay under the mouse pointer".
        A simpler approach is just to move the sub-cube by some factor of dx, dy.
        """
        if self.picked_id is None:
            return
        sub_cube = self.renderer.data.get_sub_cube_by_index(self.picked_id)
        if not sub_cube:
            return

        # We can move in the camera's plane. 
        # Let's take the camera's right and up vectors from its orientation.
        # But in your camera you have yaw/pitch. We can compute them or store them.
        # For simplicity, let's do naive XY plane translation:
        move_speed = 0.01
        tx = dx * move_speed
        ty = -dy * move_speed

        trans_mat = glm.translate(glm.mat4(1.0), glm.vec3(tx, ty, 0.0))
        sub_cube.model_matrix = trans_mat * sub_cube.model_matrix

    def cache_view_proj(self, view, proj):
        """
        So that we can use the correct MVP in the pick pass.
        Call this from your main loop right before the normal draw.
        """
        self.cached_view = view
        self.cached_proj = proj
