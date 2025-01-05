import glfw
from OpenGL.GL import *
import glm
import os

# Local imports
from camera import Camera
from data_structures import RubiksData
from rubik import RubiksCubeRenderer
from solver import RubikSolver
from picking import ColorPickingManager

##############################
# Global Variables
##############################

WIN_WIDTH = 800
WIN_HEIGHT = 800

camera = None
picking_manager = None
rubiks_renderer = None

##############################
# Callbacks
##############################

def mouse_button_callback(window, button, action, mods):
    global picking_manager, rubiks_renderer

    # Left or right click pressed => do picking if picking mode is on
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        # In picking mode, pick a sub-cube. Otherwise do normal camera orbit.
        if picking_manager.enabled:
            x, y = glfw.get_cursor_pos(window)
            picking_manager.pick(window, int(x), int(y), WIN_WIDTH, WIN_HEIGHT)
        else:
            print("MOUSE LEFT CLICK (rotate camera or if face rotation in assignment)")

    elif button == glfw.MOUSE_BUTTON_RIGHT and action == glfw.PRESS:
        if picking_manager.enabled:
            x, y = glfw.get_cursor_pos(window)
            picking_manager.pick(window, int(x), int(y), WIN_WIDTH, WIN_HEIGHT)
        else:
            print("MOUSE RIGHT CLICK (panning camera)")

def cursor_position_callback(window, xpos, ypos):
    global camera, picking_manager

    if not picking_manager.enabled:
        # Normal camera orbit/pan
        camera.process_mouse_motion(window, xpos, ypos)
    else:
        # If a sub-cube is picked and user holds left => rotate that cube
        # If a sub-cube is picked and user holds right => translate that cube
        dx = xpos - camera.last_mouse_x
        dy = ypos - camera.last_mouse_y
        camera.last_mouse_x = xpos
        camera.last_mouse_y = ypos

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            picking_manager.rotate_picked_cube(dx, -dy, camera)
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            picking_manager.translate_picked_cube(dx, dy, camera)

def scroll_callback(window, xoffset, yoffset):
    global camera
    camera.process_scroll(yoffset)

def key_callback(window, key, scancode, action, mods):
    global rubiks_renderer, picking_manager

    if action == glfw.PRESS or action == glfw.REPEAT:
        # Quit with Esc
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        
        # Face rotations
        elif key == glfw.KEY_R:
            rubiks_renderer.rotate_face('R', rubiks_renderer.clockwise)
        elif key == glfw.KEY_L:
            rubiks_renderer.rotate_face('L', rubiks_renderer.clockwise)
        elif key == glfw.KEY_U:
            rubiks_renderer.rotate_face('U', rubiks_renderer.clockwise)
        elif key == glfw.KEY_D:
            rubiks_renderer.rotate_face('D', rubiks_renderer.clockwise)
        elif key == glfw.KEY_F:
            rubiks_renderer.rotate_face('F', rubiks_renderer.clockwise)
        elif key == glfw.KEY_B:
            rubiks_renderer.rotate_face('B', rubiks_renderer.clockwise)

        # Flip rotation direction
        elif key == glfw.KEY_SPACE:
            rubiks_renderer.clockwise = not rubiks_renderer.clockwise
            print("Now rotation is clockwise=", rubiks_renderer.clockwise)

        # Z => Halve rotation angle
        elif key == glfw.KEY_Z:
            rubiks_renderer.rotation_angle /= 2.0
            if rubiks_renderer.rotation_angle < 45.0:
                rubiks_renderer.rotation_angle = 45.0
            print("Rotation angle:", rubiks_renderer.rotation_angle)

        # A => Double rotation angle
        elif key == glfw.KEY_A:
            rubiks_renderer.rotation_angle *= 2.0
            if rubiks_renderer.rotation_angle > 180.0:
                rubiks_renderer.rotation_angle = 180.0
            print("Rotation angle:", rubiks_renderer.rotation_angle)

        # Arrow keys => rotate entire cube around scene X/Y 
        elif key == glfw.KEY_UP:
            # Rotate all sub-cubes around global X by +10 deg, for example
            for scube in rubiks_renderer.data.sub_cubes:
                scube.rotation.x += 10.0
                scube.update_model_matrix()
        elif key == glfw.KEY_DOWN:
            for scube in rubiks_renderer.data.sub_cubes:
                scube.rotation.x -= 10.0
                scube.update_model_matrix()
        elif key == glfw.KEY_LEFT:
            for scube in rubiks_renderer.data.sub_cubes:
                scube.rotation.y -= 10.0
                scube.update_model_matrix()
        elif key == glfw.KEY_RIGHT:
            for scube in rubiks_renderer.data.sub_cubes:
                scube.rotation.y += 10.0
                scube.update_model_matrix()

        # Toggle picking mode
        elif key == glfw.KEY_P:
            picking_manager.toggle()

        # Mixer => random scramble
        elif key == glfw.KEY_M:
            rubiks_renderer.solver.random_mixer(rubiks_renderer, steps=10)

        # Solver => naive unscramble
        elif key == glfw.KEY_S:
            rubiks_renderer.solver.solve(rubiks_renderer)

def main():
    global camera, picking_manager, rubiks_renderer

    if not glfw.init():
        print("Failed to init GLFW")
        return

    # Create window
    window = glfw.create_window(WIN_WIDTH, WIN_HEIGHT, "Rubik's Cube - Full Assignment", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Callbacks
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_position_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    # Setup camera
    camera = Camera(WIN_WIDTH, WIN_HEIGHT)

    # Load/compile shaders
    shader_prog = create_shader_program(
        os.path.join("shaders", "basic.vert"),
        os.path.join("shaders", "basic.frag")
    )

    # Load texture
    texture_id = load_texture(os.path.join("textures", "plane.png"))

    # Enable depth test
    glEnable(GL_DEPTH_TEST)

    # Build Rubik’s data (default 3x3, but can set other sizes for bonus)
    rubiks_data = RubiksData(size=3)

    # Create solver
    from solver import RubikSolver
    solver = RubikSolver()

    # Create the renderer
    rubiks_renderer = RubiksCubeRenderer(rubiks_data, shader_prog, texture_id, solver)

    # Create picking manager
    from picking import ColorPickingManager
    picking_manager = ColorPickingManager(rubiks_renderer)

    # Main loop
    while not glfw.window_should_close(window):
        glViewport(0, 0, WIN_WIDTH, WIN_HEIGHT)
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Build camera transforms
        view = camera.get_view_matrix()
        proj = camera.get_perspective_matrix()

        # Draw the rubik’s
        rubiks_renderer.draw(view, proj)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

def create_shader_program(vertex_path, fragment_path):
    vertex_code = open(vertex_path, 'r').read()
    fragment_code = open(fragment_path, 'r').read()

    vs = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vs, vertex_code)
    glCompileShader(vs)
    if glGetShaderiv(vs, GL_COMPILE_STATUS) != GL_TRUE:
        print(glGetShaderInfoLog(vs))
        raise RuntimeError("Vertex shader compilation failed")

    fs = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fs, fragment_code)
    glCompileShader(fs)
    if glGetShaderiv(fs, GL_COMPILE_STATUS) != GL_TRUE:
        print(glGetShaderInfoLog(fs))
        raise RuntimeError("Fragment shader compilation failed")

    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)

    if glGetProgramiv(program, GL_LINK_STATUS) != GL_TRUE:
        print(glGetProgramInfoLog(program))
        raise RuntimeError("Shader program linking failed")

    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

def load_texture(path):
    from PIL import Image
    image = Image.open(path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert("RGBA").tobytes()
    width, height = image.size

    tex_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)
    return tex_id

if __name__ == "__main__":
    main()
