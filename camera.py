import glm
import glfw
import math

class Camera:
    """
    Camera with perspective, rotating around a configurable center point.
    Handles input for rotation (via mouse), zoom (scroll), and panning (right drag).
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height

        # Perspective attributes
        self.fov = 45.0
        self.near = 0.1
        self.far = 100.0

        # Position on a sphere, looking at a configurable center
        self.radius = 8.0
        self.yaw = 0.0
        self.pitch = 0.0

        # Center of rotation (can be moved with arrow keys)
        self.center = glm.vec3(0.0, 0.0, 0.0)
        self.step_size = 1.1  # Same as cube piece spacing

        # For panning in the XY plane
        self.panX = 0.0
        self.panY = 0.0

        # For storing last mouse position
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

    def get_view_matrix(self):
        # Convert yaw/pitch/radius to cartesian
        yaw_rad   = glm.radians(self.yaw)
        pitch_rad = glm.radians(self.pitch)

        # Limit pitch so you don't go upside down
        if self.pitch > 89.9:  
            self.pitch = 89.9
        if self.pitch < -89.9:
            self.pitch = -89.9

        x = self.radius * glm.cos(pitch_rad) * glm.sin(yaw_rad)
        y = self.radius * glm.sin(pitch_rad)
        z = self.radius * glm.cos(pitch_rad) * glm.cos(yaw_rad)

        eye = glm.vec3(x + self.panX, y + self.panY, z)
        center = self.center  # Use the configurable center
        up = glm.vec3(0.0, 1.0, 0.0)
        return glm.lookAt(eye, center, up)

    def get_perspective_matrix(self):
        aspect = self.width / float(self.height)
        return glm.perspective(glm.radians(self.fov), aspect, self.near, self.far)

    def process_mouse_motion(self, window, x, y):
        dx = x - self.last_mouse_x
        dy = y - self.last_mouse_y
        self.last_mouse_x = x
        self.last_mouse_y = y

        # Left button = orbit
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.yaw   += dx * 0.3
            self.pitch -= dy * 0.3
        # Right button = pan
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            self.panX += dx * 0.01
            self.panY -= dy * 0.01

    def process_scroll(self, yoffset):
        # Zoom in/out by changing radius
        self.radius -= yoffset
        if self.radius < 2.0:
            self.radius = 2.0
        if self.radius > 40.0:
            self.radius = 40.0
            
    def move_center(self, direction, axis, N):
        """
        Move the center of rotation along the specified axis.
        direction: 1 or -1
        axis: 'x', 'y', or 'z'
        """
        if axis == 'x':
            self.center.x += direction * self.step_size
        elif axis == 'y':
            self.center.y += direction * self.step_size
        elif axis == 'z':
            self.center.z += direction * self.step_size

        # Keep the center within the bounds of the cube
        step = 1.1  # Same as the cube piece spacing
        max_offset = ((N - 1) / 2.0) * step
        self.center.x = max(min(self.center.x, max_offset), -max_offset)
        self.center.y = max(min(self.center.y, max_offset), -max_offset)
        self.center.z = max(min(self.center.z, max_offset), -max_offset)