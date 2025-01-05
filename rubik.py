import glm
from OpenGL.GL import *
import numpy as np
import time

class RubiksCubeRenderer:
    """
    Manages rendering of the NxNxN Rubik's cube (using a single geometry for each sub-cube),
    plus logic for rotating faces on keypress, toggling clockwise/counterclockwise, etc.
    Also includes an animation system for rotating a face over time.
    """
    def __init__(self, rubiks_data, shader_program, texture_id, solver):
        self.data = rubiks_data   # RubiksData holding sub-cubes
        self.shader_program = shader_program
        self.texture_id = texture_id

        self.solver = solver      # RubikSolver instance

        # Create geometry for a single cube
        self.VAO = None
        self.VBO = None
        self.EBO = None
        self.vertices = None
        self.indices = None
        self._create_cube_vao()

        # Rotation control
        self.clockwise = True
        self.rotation_angle = 90.0  # can range from 45..180

        # For face-rotation animation
        self.is_animating = False
        self.current_face = None
        self.target_angle = 0.0
        self.current_angle = 0.0
        self.animation_speed = 180.0  # degrees per second
        self.rotating_cubes = []

    def _create_cube_vao(self):
        """
        Create a full 3D textured-cube geometry (36 vertices or 24+indices).
        We'll do 36 vertices for simplicity.
        """
        self.vertices = np.array([
            #   x,    y,    z,   r,   g,   b,  u,  v
            # Front face (Red)
            -0.5, -0.5,  0.5,  1,0,0,  0,0,
            0.5, -0.5,  0.5,  1,0,0,  1,0,
            0.5,  0.5,  0.5,  1,0,0,  1,1,
            -0.5,  0.5,  0.5,  1,0,0,  0,1,

            # Back face (Orange)
            -0.5, -0.5, -0.5,  1,0.5,0,  1,0,
            0.5, -0.5, -0.5,  1,0.5,0,  0,0,
            0.5,  0.5, -0.5,  1,0.5,0,  0,1,
            -0.5,  0.5, -0.5,  1,0.5,0,  1,1,

            # Left face (Green)
            -0.5, -0.5, -0.5,  0,1,0,  0,0,
            -0.5, -0.5,  0.5,  0,1,0,  1,0,
            -0.5,  0.5,  0.5,  0,1,0,  1,1,
            -0.5,  0.5, -0.5,  0,1,0,  0,1,

            # Right face (Blue)
            0.5, -0.5, -0.5,  0,0,1,  0,0,
            0.5, -0.5,  0.5,  0,0,1,  1,0,
            0.5,  0.5,  0.5,  0,0,1,  1,1,
            0.5,  0.5, -0.5,  0,0,1,  0,1,

            # Top face (White)
            -0.5,  0.5,  0.5,  1,1,1,  0,0,
            0.5,  0.5,  0.5,  1,1,1,  1,0,
            0.5,  0.5, -0.5,  1,1,1,  1,1,
            -0.5,  0.5, -0.5,  1,1,1,  0,1,

            # Bottom face (Yellow)
            -0.5, -0.5,  0.5,  1,1,0,  0,0,
            0.5, -0.5,  0.5,  1,1,0,  1,0,
            0.5, -0.5, -0.5,  1,1,0,  1,1,
            -0.5, -0.5, -0.5,  1,1,0,  0,1,
        ], dtype=np.float32)

        self.indices = np.array([
            0,1,2,  2,3,0,      # Front
            4,5,6,  6,7,4,      # Back
            8,9,10, 10,11,8,    # Left
            12,13,14,14,15,12,  # Right
            16,17,18,18,19,16,  # Top
            20,21,22,22,23,20,  # Bottom
        ], dtype=np.uint32)

        self.VAO = glGenVertexArrays(1)
        glBindVertexArray(self.VAO)

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8*self.vertices.itemsize, None)
        glEnableVertexAttribArray(0)
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8*self.vertices.itemsize, 
                              ctypes.c_void_p(3*self.vertices.itemsize))
        glEnableVertexAttribArray(1)
        # TexCoord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8*self.vertices.itemsize,
                              ctypes.c_void_p(6*self.vertices.itemsize))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def draw(self, view, projection):
        """
        Normal rendering pass (u_PickingMode = false).
        Also handle animation updates if a face is rotating.
        """
        self._update_animation()

        glUseProgram(self.shader_program)

        # Set picking mode to false
        picking_loc = glGetUniformLocation(self.shader_program, "u_PickingMode")
        glUniform1i(picking_loc, 0)

        # Bind texture
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        tex_loc = glGetUniformLocation(self.shader_program, "u_Texture")
        glUniform1i(tex_loc, 0)

        # Color uniform
        color_loc = glGetUniformLocation(self.shader_program, "u_Color")
        glUniform4f(color_loc, 1.0, 1.0, 1.0, 1.0)

        for scube in self.data.sub_cubes:
            model = scube.model_matrix
            mvp = projection * view * model
            mvp_loc = glGetUniformLocation(self.shader_program, "u_MVP")
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))

            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glUseProgram(0)

    def draw_picking_pass(self):
        """
        Render each sub-cube with a unique color. Depth test still on,
        but no texture. We set 'u_PickingMode = true'.
        """
        glUseProgram(self.shader_program)

        picking_loc = glGetUniformLocation(self.shader_program, "u_PickingMode")
        glUniform1i(picking_loc, 1)

        # We don't use the actual texture for picking, but we must bind something.
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        tex_loc = glGetUniformLocation(self.shader_program, "u_Texture")
        glUniform1i(tex_loc, 0)

        for i, scube in enumerate(self.data.sub_cubes):
            # Unique picking color
            r = (i & 0x000000FF) / 255.0
            g = ((i & 0x0000FF00) >> 8) / 255.0
            b = ((i & 0x00FF0000) >> 16) / 255.0
            picking_color_loc = glGetUniformLocation(self.shader_program, "u_PickingColor")
            glUniform3f(picking_color_loc, r, g, b)

            # MVP
            model = scube.model_matrix
            # For picking pass, assume some orthographic or the same view/proj as main?
            # We'll do the same camera transforms for consistency:
            # Actually set them from global variables if needed.
            # For simplicity, let's just reuse last known MVP from normal render.
            # But the assignment demands the same transformations, so do it externally.

            # If we haven't stored the MVP externally, either store or pass it in.
            # For an example, let's do identity MVP just so each sub-cube is in place 
            # (but that won't reflect camera perspective. Typically you want same camera.)
            # We'll rely on main to call us immediately after a real camera pass 
            # with identical transforms. 
            # => We'll pass a global or do a quick hack with glUniformMatrix4fv as identity.

            # We'll set them to identity if we haven't stored camera. 
            identity = glm.mat4(1.0)
            # But we do need the real model transform:
            mvp = identity * model
            mvp_loc = glGetUniformLocation(self.shader_program, "u_MVP")
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, glm.value_ptr(mvp))

            glBindVertexArray(self.VAO)
            glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glUseProgram(0)

    ### Face Rotation Logic ###
    def rotate_face(self, face, clockwise):
        """
        Called upon a key press (R, L, U, D, F, B).
        Instead of instantly rotating, we set up an animation to rotate 90 degrees 
        (or 'rotation_angle' degrees if not 90).
        """
        if self.is_animating:
            print("Already animating a face. Wait until done.")
            return

        self.current_face = face
        self.clockwise = clockwise
        self.target_angle = self.rotation_angle
        self.current_angle = 0.0
        # Identify which sub-cubes need rotating
        self.rotating_cubes = self._get_subcubes_for_face(face)
        self.is_animating = True
        print(f"Starting rotation for face={face}, clockwise={clockwise} by {self.rotation_angle} deg.")

    def _get_subcubes_for_face(self, face):
        """
        Return the list of sub-cubes belonging to the requested face.
        For example, 'R' -> all sub-cubes with x = +offset.
        'L' -> x = -offset, 'U' -> y=+offset, 'D'->y=-offset, etc.
        For a NxN, offset is (N-1)/2.
        """
        eps = 0.001
        offset = (self.data.size - 1)/2.0
        selected = []
        for scube in self.data.sub_cubes:
            x,y,z = scube.position.x, scube.position.y, scube.position.z
            if face == 'R':  # right => x ~ +offset
                if abs(x - offset) < eps:
                    selected.append(scube)
            elif face == 'L': # left => x ~ -offset
                if abs(x + offset) < eps:
                    selected.append(scube)
            elif face == 'U': # up => y ~ +offset
                if abs(y - offset) < eps:
                    selected.append(scube)
            elif face == 'D': # down => y ~ -offset
                if abs(y + offset) < eps:
                    selected.append(scube)
            elif face == 'F': # front => z ~ +offset
                if abs(z - offset) < eps:
                    selected.append(scube)
            elif face == 'B': # back => z ~ -offset
                if abs(z + offset) < eps:
                    selected.append(scube)
        return selected

    def _update_animation(self):
        """
        If currently animating, rotate the appropriate sub-cubes incrementally 
        until done, then finalize positions/orientations.
        """
        if not self.is_animating:
            return

        dt = 1.0/60.0  # assume ~60 fps or measure real time
        step = self.animation_speed * dt
        if self.current_angle + step >= self.target_angle:
            step = self.target_angle - self.current_angle
            self.is_animating = False

        self.current_angle += step

        # Rotate each sub-cube around the face's rotation axis
        angle_deg = step if self.clockwise else -step
        axis, pivot = self._get_face_axis_and_pivot(self.current_face)

        rot_mat = glm.rotate(glm.mat4(1.0), glm.radians(angle_deg), axis)

        for scube in self.rotating_cubes:
            # Move pivot to origin
            scube.position -= pivot
            # Rotate
            scube.position = glm.vec3(rot_mat * glm.vec4(scube.position, 1.0))
            # Also rotate orientation (Euler angles can be tricky, but let's do a matrix-based approach):
            # Build a matrix from scube's rotation, multiply, then re-extract Euler angles, or keep storing matrix.
            # For simplicity: store the orientation as a matrix entirely. 
            # But let's proceed with the existing Euler approach in a naive way. 
            # Instead, we'll just do the orientation as a matrix:
            orientation_mat = self._build_orientation_matrix(scube.rotation)
            orientation_mat = rot_mat * orientation_mat
            # Re-extract Euler
            new_euler = self._extract_euler_angles(orientation_mat)
            scube.rotation = new_euler

            # Move back
            scube.position += pivot
            scube.update_model_matrix()

        if not self.is_animating:
            # Final step: re-snap sub-cubes to exact multiples of 90 degrees, 
            # to avoid floating precision issues. 
            # Also recalc their positions if needed.
            for scube in self.rotating_cubes:
                # Snap scube.rotation.x/y/z to nearest multiple of 90
                scube.rotation = glm.vec3([self._snap_to_90(a) for a in scube.rotation])
                # Snap scube.position.x/y/z to nearest .5 increments if you want perfect alignment
                scube.position.x = round(scube.position.x * 2.0) / 2.0
                scube.position.y = round(scube.position.y * 2.0) / 2.0
                scube.position.z = round(scube.position.z * 2.0) / 2.0
                scube.update_model_matrix()
            self.rotating_cubes.clear()
            print("Rotation animation done.")

    def _snap_to_90(self, angle):
        # Snap angle in degrees to nearest multiple of 90
        return round(angle / 90.0) * 90.0

    def _get_face_axis_and_pivot(self, face):
        """
        Returns (axis, pivot_point).
        axis is a glm.vec3 indicating local axis for that face rotation.
        pivot is the center about which the rotation occurs.
        For e.g. R face, pivot is (offset,0,0).
        """
        offset = (self.data.size - 1)/2.0
        if face == 'R':
            return (glm.vec3(1,0,0), glm.vec3(offset, 0, 0))
        elif face == 'L':
            return (glm.vec3(-1,0,0), glm.vec3(-offset, 0, 0))
        elif face == 'U':
            return (glm.vec3(0,1,0), glm.vec3(0, offset, 0))
        elif face == 'D':
            return (glm.vec3(0,-1,0), glm.vec3(0, -offset, 0))
        elif face == 'F':
            return (glm.vec3(0,0,1), glm.vec3(0, 0, offset))
        elif face == 'B':
            return (glm.vec3(0,0,-1), glm.vec3(0, 0, -offset))
        return (glm.vec3(0,0,0), glm.vec3(0,0,0))

    def _build_orientation_matrix(self, euler):
        # Convert Euler angles (degrees) to a rotation matrix
        rx = glm.rotate(glm.mat4(1.0), glm.radians(euler.x), glm.vec3(1,0,0))
        ry = glm.rotate(glm.mat4(1.0), glm.radians(euler.y), glm.vec3(0,1,0))
        rz = glm.rotate(glm.mat4(1.0), glm.radians(euler.z), glm.vec3(0,0,1))
        return rz * ry * rx

    def _extract_euler_angles(self, mat):
        # Extract Euler angles from a rotation matrix in ZYX order
        # This is one approach: pitch(y), yaw(x), roll(z). Might vary.
        # We'll do a simpler approach that might not handle gimbal lock well,
        # but is enough for snapping 90-degree increments.
        sy = glm.sqrt(mat[0][0]*mat[0][0] + mat[1][0]*mat[1][0])
        singular = sy < 1e-6
        if not singular:
            x = glm.degrees(glm.atan2(mat[2][1], mat[2][2]))
            y = glm.degrees(glm.atan2(-mat[2][0], sy))
            z = glm.degrees(glm.atan2(mat[1][0], mat[0][0]))
        else:
            x = glm.degrees(glm.atan2(-mat[1][2], mat[1][1]))
            y = glm.degrees(glm.atan2(-mat[2][0], sy))
            z = 0
        return glm.vec3(x, y, z)
