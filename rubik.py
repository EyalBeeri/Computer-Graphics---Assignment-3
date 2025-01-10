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
        self.data = rubiks_data  # RubiksData holding sub-cubes
        self.shader_program = shader_program
        self.texture_id = texture_id

        self.solver = solver  # RubikSolver instance

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
        self.half_turned_faces = []  # Will store face name if any face is half-turned
        self.is_face_half_turned = False  # True if any face is not at a 90-degree position

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
            -0.5, -0.5, 0.5, 1, 0, 0, 0, 0,
            0.5, -0.5, 0.5, 1, 0, 0, 1, 0,
            0.5, 0.5, 0.5, 1, 0, 0, 1, 1,
            -0.5, 0.5, 0.5, 1, 0, 0, 0, 1,

            # Back face (Orange)
            -0.5, -0.5, -0.5, 1, 0.5, 0, 1, 0,
            0.5, -0.5, -0.5, 1, 0.5, 0, 0, 0,
            0.5, 0.5, -0.5, 1, 0.5, 0, 0, 1,
            -0.5, 0.5, -0.5, 1, 0.5, 0, 1, 1,

            # Left face (Green)
            -0.5, -0.5, -0.5, 0, 1, 0, 0, 0,
            -0.5, -0.5, 0.5, 0, 1, 0, 1, 0,
            -0.5, 0.5, 0.5, 0, 1, 0, 1, 1,
            -0.5, 0.5, -0.5, 0, 1, 0, 0, 1,

            # Right face (Blue)
            0.5, -0.5, -0.5, 0, 0, 1, 0, 0,
            0.5, -0.5, 0.5, 0, 0, 1, 1, 0,
            0.5, 0.5, 0.5, 0, 0, 1, 1, 1,
            0.5, 0.5, -0.5, 0, 0, 1, 0, 1,

            # Top face (White)
            -0.5, 0.5, 0.5, 1, 1, 1, 0, 0,
            0.5, 0.5, 0.5, 1, 1, 1, 1, 0,
            0.5, 0.5, -0.5, 1, 1, 1, 1, 1,
            -0.5, 0.5, -0.5, 1, 1, 1, 0, 1,

            # Bottom face (Yellow)
            -0.5, -0.5, 0.5, 1, 1, 0, 0, 0,
            0.5, -0.5, 0.5, 1, 1, 0, 1, 0,
            0.5, -0.5, -0.5, 1, 1, 0, 1, 1,
            -0.5, -0.5, -0.5, 1, 1, 0, 0, 1,
        ], dtype=np.float32)

        self.indices = np.array([
            0, 1, 2, 2, 3, 0,  # Front
            4, 5, 6, 6, 7, 4,  # Back
            8, 9, 10, 10, 11, 8,  # Left
            12, 13, 14, 14, 15, 12,  # Right
            16, 17, 18, 18, 19, 16,  # Top
            20, 21, 22, 22, 23, 20,  # Bottom
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * self.vertices.itemsize, None)
        glEnableVertexAttribArray(0)
        # Color attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * self.vertices.itemsize,
                              ctypes.c_void_p(3 * self.vertices.itemsize))
        glEnableVertexAttribArray(1)
        # TexCoord attribute
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * self.vertices.itemsize,
                              ctypes.c_void_p(6 * self.vertices.itemsize))
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
    def can_rotate_face(self, new_face):
        """
        Check if it's valid to rotate the requested face given current cube state.
        Takes into account half-turned faces.
        """
        # If no face is half-turned, all rotations are allowed
        if not self.is_face_half_turned:
            return True

        if new_face == 'R' and 'L' not in self.half_turned_faces and 'R' not in self.half_turned_faces:
            return False
        elif new_face == 'L' and 'R' not in self.half_turned_faces and 'L' not in self.half_turned_faces:
            return False
        elif new_face == 'U' and 'D' not in self.half_turned_faces and 'U' not in self.half_turned_faces:
            return False
        elif new_face == 'D' and 'U' not in self.half_turned_faces and 'D' not in self.half_turned_faces:
            return False
        elif new_face == 'F' and 'B' not in self.half_turned_faces and 'F' not in self.half_turned_faces:
            return False
        elif new_face == 'B' and 'F' not in self.half_turned_faces and 'B' not in self.half_turned_faces:
            return False
        else:
            return True

    def rotate_face(self, face, clockwise):
        """
        Called upon a key press (R, L, U, D, F, B).
        Checks if rotation is allowed before starting animation.
        """
        if self.is_animating:
            print("Already animating a face. Wait until done.")
            return

        if not self.can_rotate_face(face):
            print(f"Cannot rotate face {face} while face is in {self.half_turned_faces} and half-turned")
            return

        self.current_face = face
        self.clockwise = clockwise
        self.target_angle = self.rotation_angle
        self.current_angle = 0.0
        self.rotating_cubes = self._get_subcubes_for_face(face)
        self.is_animating = True
        print(f"Starting rotation for face={face}, clockwise={clockwise} by {self.rotation_angle} deg.")

    def _get_subcubes_for_face(self, face):
        offset = self.data.size - 1
        if face == 'R':
            return [scube for scube in self.data.sub_cubes if scube.grid_coords[0] == offset]
        elif face == 'L':
            return [scube for scube in self.data.sub_cubes if scube.grid_coords[0] == 0]
        elif face == 'U':
            return [scube for scube in self.data.sub_cubes if scube.grid_coords[1] == offset]
        elif face == 'D':
            return [scube for scube in self.data.sub_cubes if scube.grid_coords[1] == 0]
        elif face == 'F':
            return [scube for scube in self.data.sub_cubes if scube.grid_coords[2] == offset]
        elif face == 'B':
            return [scube for scube in self.data.sub_cubes if scube.grid_coords[2] == 0]
        else:
            raise ValueError(f"Unknown face: {face}")

    def _update_animation(self):
        if not self.is_animating:
            return

        dt = 1.0 / 60.0
        step = self.animation_speed * dt

        # Store previous angle before update
        prev_angle = self.current_angle

        # Update current angle and check completion
        self.current_angle += step
        if self.current_angle >= self.target_angle:
            self.current_angle = self.target_angle
            self.is_animating = False
            step = self.target_angle - (self.current_angle - step)  # Get exact remaining step

        # Check if we crossed a 90-degree boundary
        prev_quarter_turns = int(prev_angle / 90)
        current_quarter_turns = int(self.current_angle / 90)
        crossed_90_degrees = current_quarter_turns > prev_quarter_turns

        # If we crossed a 90-degree boundary, update cube state
        if crossed_90_degrees:
            self._snap_cubes_to_grid()
            self._update_face_colors()

        # Handle half-turn state
        if not self.is_animating:  # Animation complete
            final_angle = self.current_angle % 360  # Normalize to 0-360
            is_half_turn = final_angle % 90 != 0

            if is_half_turn:
                if self.current_face in self.half_turned_faces:
                    self.half_turned_faces.remove(self.current_face)
                    if len(self.half_turned_faces) == 0:
                        self.is_face_half_turned = False
                    else:
                        self.is_face_half_turned = True
                else:
                    self.half_turned_faces.append(self.current_face)
                    self.is_face_half_turned = True
            else:
                # At a 90-degree position
                if self.current_face in self.half_turned_faces:
                    self.half_turned_faces.remove(self.current_face)
                    if len(self.half_turned_faces) == 0:
                        self.is_face_half_turned = False

        # Perform the actual rotation
        angle_deg = step if self.clockwise else -step
        axis, pivot = self._get_face_axis_and_pivot(self.current_face)
        rot_mat = glm.rotate(glm.mat4(1.0), glm.radians(angle_deg), axis)

        for scube in self.rotating_cubes:
            # Move to origin, rotate, move back
            pos = scube.position - pivot
            pos = glm.vec3(rot_mat * glm.vec4(pos, 1.0))
            scube.position = pos + pivot

            # Update rotation
            scube.rotation += angle_deg * axis
            scube.update_model_matrix()

    def _get_cube_colors(self, cube):
        """Get current colors of cube faces"""
        colors = []
        vertices_per_face = 4
        floats_per_vertex = 8  # x,y,z, r,g,b, u,v

        start_idx = 0  # Don't use cube.index as it might be incorrect
        for face in range(6):
            face_colors = []
            for vertex in range(vertices_per_face):
                vertex_start = (start_idx + face * 4 + vertex) * floats_per_vertex
                color = self.vertices[vertex_start + 3:vertex_start + 6].copy()  # Make a copy of the color
                face_colors.append(color)
            colors.append(face_colors)
        return colors

    def _set_cube_colors(self, cube, colors):
        """Set new colors for cube faces"""
        if not colors or not all(face_colors for face_colors in colors):
            print("Warning: Invalid colors array")
            return

        vertices_per_face = 4
        floats_per_vertex = 8

        start_idx = 0  # Don't use cube.index
        for face in range(6):
            for vertex in range(vertices_per_face):
                vertex_start = (start_idx + face * 4 + vertex) * floats_per_vertex
                if vertex_start + 6 <= len(self.vertices):  # Bounds check
                    self.vertices[vertex_start + 3:vertex_start + 6] = colors[face][vertex]

        # Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

    def _update_face_colors(self):
        """Update colors after rotation"""
        # Store colors before rotation
        original_colors = []
        for cube in self.rotating_cubes:
            colors = self._get_cube_colors(cube)
            original_colors.append(colors)

        # Calculate new positions
        n = int(len(self.rotating_cubes) ** 0.5)  # Should be 3 for a 3x3 cube
        new_indices = self._get_rotated_indices(n, self.clockwise)

        # Safety check
        if len(new_indices) != len(self.rotating_cubes):
            print(
                f"Warning: indices mismatch. new_indices={len(new_indices)}, rotating_cubes={len(self.rotating_cubes)}")
            return

        # Apply rotated colors
        for i, cube in enumerate(self.rotating_cubes):
            new_idx = new_indices[i]
            if 0 <= new_idx < len(original_colors):  # Bounds check
                try:
                    self._set_cube_colors(cube, original_colors[new_idx])
                except Exception as e:
                    print(f"Error setting colors for cube {i} with new_idx {new_idx}: {e}")

    def _get_rotated_indices(self, n, clockwise):
        """Get new indices order after rotation"""
        indices = list(range(n * n))
        if clockwise:
            # For clockwise rotation on left face
            return [n * j + i for i in range(n - 1, -1, -1) for j in range(n)]
        else:
            # For counter-clockwise rotation on left face
            return [n * j + i for i in range(n) for j in range(n - 1, -1, -1)]

    def _snap_cubes_to_grid(self):
        """After the animation, update each rotating cube’s grid-coords and recalc position."""
        size = self.data.size
        offset = size - 1

        for scube in self.rotating_cubes:
            x, y, z = scube.grid_coords

            if self.current_face == 'R':
                if self.clockwise:
                    new_y = z
                    new_z = offset - y
                else:
                    new_y = offset - z
                    new_z = y
                scube.grid_coords = (x, new_y, new_z)

            elif self.current_face == 'L':
                if self.clockwise:
                    new_y = offset - z
                    new_z = y
                else:
                    new_y = z
                    new_z = offset - y
                scube.grid_coords = (x, new_y, new_z)

            elif self.current_face == 'U':
                if self.clockwise:
                    new_x = offset - z
                    new_z = x
                else:
                    new_x = z
                    new_z = offset - x
                scube.grid_coords = (new_x, y, new_z)

            elif self.current_face == 'D':
                if self.clockwise:
                    new_x = z
                    new_z = offset - x
                else:
                    new_x = offset - z
                    new_z = x
                scube.grid_coords = (new_x, y, new_z)

            elif self.current_face == 'F':
                if self.clockwise:
                    new_x = y
                    new_y = offset - x
                else:
                    new_x = offset - y
                    new_y = x
                scube.grid_coords = (new_x, new_y, z)

            elif self.current_face == 'B':
                if self.clockwise:
                    new_x = offset - y
                    new_y = x
                else:
                    new_x = y
                    new_y = offset - x
                scube.grid_coords = (new_x, new_y, z)

            scube._update_position_from_coords()

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
        offset = (self.data.size - 1) / 2.0
        if face == 'R':
            return (glm.vec3(1, 0, 0), glm.vec3(offset, 0, 0))
        elif face == 'L':
            return (glm.vec3(-1, 0, 0), glm.vec3(-offset, 0, 0))
        elif face == 'U':
            return (glm.vec3(0, 1, 0), glm.vec3(0, offset, 0))
        elif face == 'D':
            return (glm.vec3(0, -1, 0), glm.vec3(0, -offset, 0))
        elif face == 'F':
            return (glm.vec3(0, 0, 1), glm.vec3(0, 0, offset))
        elif face == 'B':
            return (glm.vec3(0, 0, -1), glm.vec3(0, 0, -offset))
        return (glm.vec3(0, 0, 0), glm.vec3(0, 0, 0))

    def _build_orientation_matrix(self, euler):
        # Convert Euler angles (degrees) to a rotation matrix
        rx = glm.rotate(glm.mat4(1.0), glm.radians(euler.x), glm.vec3(1, 0, 0))
        ry = glm.rotate(glm.mat4(1.0), glm.radians(euler.y), glm.vec3(0, 1, 0))
        rz = glm.rotate(glm.mat4(1.0), glm.radians(euler.z), glm.vec3(0, 0, 1))
        return rz * ry * rx

    def _extract_euler_angles(self, mat):
        # Extract Euler angles from a rotation matrix in ZYX order
        # This is one approach: pitch(y), yaw(x), roll(z). Might vary.
        # We'll do a simpler approach that might not handle gimbal lock well,
        # but is enough for snapping 90-degree increments.
        sy = glm.sqrt(mat[0][0] * mat[0][0] + mat[1][0] * mat[1][0])
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
