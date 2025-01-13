import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
from PIL import Image
import glm
from camera import Camera
from rubiks_cube_state import RubiksCubeController


class RubiksCube:
    def __init__(self, width=800, height=600):
        self.camera = Camera(width, height)
        self.cube_controller = RubiksCubeController()

        with open('shaders/basic.vert', 'r') as f:
            self.VERTEX_SHADER = f.read()
        with open('shaders/basic.frag', 'r') as f:
            self.FRAGMENT_SHADER = f.read()

        self.COLORS = {
            'white': [1.0, 1.0, 1.0],
            'yellow': [1.0, 1.0, 0.0],
            'red': [1.0, 0.0, 0.0],
            'orange': [1.0, 0.5, 0.0],
            'green': [0.0, 1.0, 0.0],
            'blue': [0.0, 0.0, 1.0]
        }

        # Same vertices as before but with texture coordinates
        self.vertices = np.array([
            # Format: x, y, z, r, g, b, tex_x, tex_y
            # Front face (white)
            -0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
            0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0,
            -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0,

            # Back face (yellow)
            -0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 1.0,
            0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 1.0,
            0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 0.0,
            -0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 0.0,

            # Right face (red)
            0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
            0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,

            # Left face (orange)
            -0.5, -0.5, -0.5, 1.0, 0.5, 0.0, 0.0, 1.0,
            -0.5, 0.5, -0.5, 1.0, 0.5, 0.0, 1.0, 1.0,
            -0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.0,
            -0.5, -0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0,

            # Top face (green)
            -0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
            0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0,
            0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
            -0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0,

            # Bottom face (blue)
            -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0,
            0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
            0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
            -0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            4, 5, 6, 6, 7, 4,  # Back
            8, 9, 10, 10, 11, 8,  # Right
            12, 13, 14, 14, 15, 12,  # Left
            16, 17, 18, 18, 19, 16,  # Top
            20, 21, 22, 22, 23, 20  # Bottom
        ], dtype=np.uint32)

        self.shader = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.texture = None

    def load_texture(self, path):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        # Set texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

        # Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Load image
        try:
            image = Image.open(path)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_data = image.tobytes()
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
            glGenerateMipmap(GL_TEXTURE_2D)
            print(f"Texture loaded successfully: {path}")
        except Exception as e:
            print(f"Error loading texture {path}: {e}")

    def init_gl(self):
        # Compile shaders
        vertex_shader = shaders.compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
        self.shader = shaders.compileProgram(vertex_shader, fragment_shader)

        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, None)
        glEnableVertexAttribArray(0)
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        # Texture coordinate attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Load texture
        self.load_texture("textures/plane.png")

        # Enable depth testing
        glEnable(GL_DEPTH_TEST)

    def draw(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.shader)

        # Set uniforms
        glUniform4f(glGetUniformLocation(self.shader, "u_Color"), 1.0, 1.0, 1.0, 1.0)
        glUniform1i(glGetUniformLocation(self.shader, "u_PickingMode"), 0)
        glUniform1i(glGetUniformLocation(self.shader, "u_Texture"), 0)

        # Activate texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        view = self.camera.get_view_matrix()
        projection = self.camera.get_perspective_matrix()

        for piece in self.cube_controller.state.pieces.values():
            # Use the piece's transformation matrix directly
            model = piece.transform

            # Calculate MVP matrix
            mvp = projection * view * model

            # Send MVP matrix to shader
            glUniformMatrix4fv(glGetUniformLocation(self.shader, "u_MVP"), 1, GL_FALSE, glm.value_ptr(mvp))

            # Draw the cube piece
            glBindVertexArray(self.vao)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)


def main():
    if not glfw.init():
        return

    window = glfw.create_window(800, 600, "Rubik's Cube", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    cube = RubiksCube(800, 600)
    cube.init_gl()

    def mouse_callback(window, xpos, ypos):
        cube.camera.process_mouse_motion(window, xpos, ypos)

    def scroll_callback(window, xoffset, yoffset):
        cube.camera.process_scroll(yoffset)

    def key_callback(window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_R:
                cube.cube_controller.process_keyboard('R')
            elif key == glfw.KEY_L:
                cube.cube_controller.process_keyboard('L')
            elif key == glfw.KEY_U:
                cube.cube_controller.process_keyboard('U')
            elif key == glfw.KEY_D:
                cube.cube_controller.process_keyboard('D')
            elif key == glfw.KEY_F:
                cube.cube_controller.process_keyboard('F')
            elif key == glfw.KEY_B:
                cube.cube_controller.process_keyboard('B')
            elif key == glfw.KEY_SPACE:
                cube.cube_controller.process_keyboard('SPACE')
            elif key == glfw.KEY_A:
                cube.cube_controller.process_keyboard('A')
            elif key == glfw.KEY_Z:
                cube.cube_controller.process_keyboard('Z')

    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    glClearColor(0.2, 0.3, 0.3, 1.0)

    while not glfw.window_should_close(window):
        cube.draw()
        glfw.swap_buffers(window)
        glfw.poll_events()


    glfw.terminate()


if __name__ == "__main__":
    main()