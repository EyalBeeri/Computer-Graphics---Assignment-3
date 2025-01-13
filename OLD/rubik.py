import numpy as np
import glm
from OpenGL.GL import *

class RubiksCubeRenderer:
    """
    Responsible for:
      - Owning the VAO, VBO for the single-cube geometry.
      - Drawing the NxNxN sub-cubes from RubiksData.
      - Handling face rotations (R, L, U, D, F, B) with immediate 90-degree turns.
    """
    def __init__(self, rubiks_data, shader_program, texture_id, solver):
        """
        :param rubiks_data: RubiksData instance
        :param shader_program: compiled shader program
        :param texture_id: loaded texture ID (plane.png)
        :param solver: RubikSolver instance
        """
        self.data = rubiks_data
        self.shader = shader_program
        self.texture_id = texture_id
        self.solver = solver

        # Rotation info
        self.rotation_angle = 90.0   # angle for each face rotation
        self.clockwise = True        # default direction (true=clockwise)

        # Build geometry (VAO, VBO, EBO) for a single cube
        self._create_cube_vao()

    def _create_cube_vao(self):
        """
        Prepare a VAO for a single cube, using the given color layout.
        We'll draw it N^3 times for an NxN Rubik’s.
        """
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        self.ebo = glGenBuffers(1)

        # A single cube has 36 indices => 12 triangles, 3 vertices each.
        # We have 24 unique vertices (6 faces x 4 corners).
        self.vertices = np.array([
            #   x,    y,    z,   r,   g,   b,  u,  v
            # Front (red)
            -0.5, -0.5,  0.5,   1, 0, 0,   0, 0,
             0.5, -0.5,  0.5,   1, 0, 0,   1, 0,
             0.5,  0.5,  0.5,   1, 0, 0,   1, 1,
            -0.5,  0.5,  0.5,   1, 0, 0,   0, 1,

            # Back (green)
            -0.5, -0.5, -0.5,   0, 1, 0,   1, 0,
             0.5, -0.5, -0.5,   0, 1, 0,   0, 0,
             0.5,  0.5, -0.5,   0, 1, 0,   0, 1,
            -0.5,  0.5, -0.5,   0, 1, 0,   1, 1,

            # Left (blue)
            -0.5, -0.5, -0.5,   0, 0, 1,   0, 0,
            -0.5, -0.5,  0.5,   0, 0, 1,   1, 0,
            -0.5,  0.5,  0.5,   0, 0, 1,   1, 1,
            -0.5,  0.5, -0.5,   0, 0, 1,   0, 1,

            # Right (yellow)
             0.5, -0.5, -0.5,   1, 1, 0,   0, 0,
             0.5, -0.5,  0.5,   1, 1, 0,   1, 0,
             0.5,  0.5,  0.5,   1, 1, 0,   1, 1,
             0.5,  0.5, -0.5,   1, 1, 0,   0, 1,

            # Top (white)
            -0.5,  0.5,  0.5,   1, 1, 1,   0, 0,
             0.5,  0.5,  0.5,   1, 1, 1,   1, 0,
             0.5,  0.5, -0.5,   1, 1, 1,   1, 1,
            -0.5,  0.5, -0.5,   1, 1, 1,   0, 1,

            # Bottom (magenta)
            -0.5, -0.5,  0.5,   1, 0, 1,   0, 0,
             0.5, -0.5,  0.5,   1, 0, 1,   1, 0,
             0.5, -0.5, -0.5,   1, 0, 1,   1, 1,
            -0.5, -0.5, -0.5,   1, 0, 1,   0, 1,
        ], dtype=np.float32)

        self.indices = np.array([
            0,  1,  2,   2,  3,  0,    # front
            4,  5,  6,   6,  7,  4,    # back
            8,  9, 10,  10, 11,  8,    # left
            12, 13, 14, 14, 15, 12,    # right
            16, 17, 18, 18, 19, 16,    # top
            20, 21, 22, 22, 23, 20     # bottom
        ], dtype=np.uint32)

        glBindVertexArray(self.vao)

        # Upload vertex data
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER,
                     self.vertices.nbytes,
                     self.vertices,
                     GL_STATIC_DRAW)

        # Upload index data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                     self.indices.nbytes,
                     self.indices,
                     GL_STATIC_DRAW)

        # Configure layout: 3 floats position, 3 floats color, 2 floats UV
        stride = 8 * self.vertices.itemsize
        # Position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(0))
        # Color
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(3 * self.vertices.itemsize))
        # UV
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE,
                              stride, ctypes.c_void_p(6 * self.vertices.itemsize))

        glBindVertexArray(0)

    def draw(self, view, proj):
        """
        Draw all sub-cubes. For each sub-cube, compute MVP, upload it, draw.
        """
        glUseProgram(self.shader)

        # Uniform locations
        loc_mvp = glGetUniformLocation(self.shader, "u_MVP")
        loc_color = glGetUniformLocation(self.shader, "u_Color")
        loc_pick = glGetUniformLocation(self.shader, "u_PickingMode")
        loc_pick_col = glGetUniformLocation(self.shader, "u_PickingColor")

        # Turn off picking mode for normal draw
        glUniform1i(loc_pick, 0)
        # Global color multiplier = white
        glUniform4f(loc_color, 1.0, 1.0, 1.0, 1.0)
        # The picking color is ignored in normal rendering
        glUniform3f(loc_pick_col, 0.0, 0.0, 0.0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        glBindVertexArray(self.vao)
        for sub_cube in self.data.sub_cubes:
            model = sub_cube.model_matrix
            mvp = proj * view * model
            glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, glm.value_ptr(mvp))
            glDrawElements(GL_TRIANGLES,
                           len(self.indices),
                           GL_UNSIGNED_INT,
                           ctypes.c_void_p(0))
        glBindVertexArray(0)

    def update_animation(self):
        """
        If we do *immediate* face rotations, we don't need 
        to animate incrementally. We'll leave this blank 
        or remove it if we want no partial rotation at all.
        """
        pass

    def rotate_face(self, face, clockwise):
        """
        Called from the key callback (R, L, U, D, F, B).
        Here we do an immediate 90° rotation. 
        That means we call _do_face_rotation(..., ±90°) once.
        """
        angle_degs = self.rotation_angle if clockwise else -self.rotation_angle
        self._do_face_rotation(face, angle_degs)

    def _do_face_rotation(self, face, angle_degrees):
        """
        Actually rotate the sub-cubes that belong to the given face by ±90°.
        Then reassign their (x,y,z) grid positions so the Rubik's doesn't break.
        """
        # Identify the axis of rotation and which coordinate = +1 / -1
        axis = glm.vec3(0, 0, 0)
        face_coord = 1

        if face == 'R':     # Right face => x=+1
            axis = glm.vec3(1, 0, 0)  
            face_coord = 1
        elif face == 'L':   # Left face => x=-1
            axis = glm.vec3(1, 0, 0)
            face_coord = -1
        elif face == 'U':   # Up face => y=+1
            axis = glm.vec3(0, 1, 0)
            face_coord = 1
        elif face == 'D':   # Down face => y=-1
            axis = glm.vec3(0, 1, 0)
            face_coord = -1
        elif face == 'F':   # Front face => z=+1
            axis = glm.vec3(0, 0, 1)
            face_coord = 1
        elif face == 'B':   # Back face => z=-1
            axis = glm.vec3(0, 0, 1)
            face_coord = -1
        else:
            return  # Unknown face

        # Gather the sub-cubes on the requested face:
        cubes_on_face = []
        for sc in self.data.sub_cubes:
            if face in ['R','L'] and sc.grid_x == face_coord:
                cubes_on_face.append(sc)
            elif face in ['U','D'] and sc.grid_y == face_coord:
                cubes_on_face.append(sc)
            elif face in ['F','B'] and sc.grid_z == face_coord:
                cubes_on_face.append(sc)

        # We'll rotate them by angle_degrees around the origin. 
        # (We placed sub-cubes in [-1,0,1] for each axis, so the origin is the center of the cube.)
        angle_rad = glm.radians(angle_degrees)
        rot_mat = glm.rotate(glm.mat4(1.0), angle_rad, axis)

        for sc in cubes_on_face:
            # 1) Update orientation
            sc.orientation = rot_mat * sc.orientation
            
            # 2) If it's exactly ±90°, finalize new integer grid coords
            if abs(angle_degrees) == 90.0:
                old_pos = glm.vec4(sc.grid_x, sc.grid_y, sc.grid_z, 1.0)
                new_pos = rot_mat * old_pos
                rx = round(new_pos.x)
                ry = round(new_pos.y)
                rz = round(new_pos.z)
                sc.update_grid_pos(rx, ry, rz)
            else:
                sc._update_model_matrix()   # partial rotations, if you wanted them

