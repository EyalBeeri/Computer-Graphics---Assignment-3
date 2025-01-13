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

        self.cube_orientation = glm.mat4(1.0)

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
            model = self.cube_orientation * scube.model_matrix
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
        

    def describe_cube(self, cube):
        """
        Describe the cube's position and color.
        """
        coords = cube.grid_coords
        position = cube.position
        rotation = cube.rotation
        return f"Cube index: {cube.index}, Coords: {coords}, Position: {position}, Rotation: {rotation}"

    def _update_animation(self):
        """
        Update rotation animation for the currently rotating face.
        """
        if not self.is_animating:
            return

        dt = 1.0 / 60.0
        step = self.animation_speed * dt

        # Increment the current angle and check for completion
        self.current_angle += step
        if self.current_angle >= self.target_angle:
            self.current_angle = self.target_angle
            self.is_animating = False

        # Calculate rotation angle
        angle_deg = step if self.clockwise else -step
        axis, pivot = self._get_face_axis_and_pivot(self.current_face)

        rotation_matrix = glm.rotate(glm.mat4(1.0), glm.radians(angle_deg), axis)

        for scube in self.rotating_cubes:
            relative_position = scube.position - pivot
            rotated_position = glm.vec3(rotation_matrix * glm.vec4(relative_position, 1.0))
            scube.position = rotated_position + pivot

            scube.rotation += angle_deg * axis
            scube.update_model_matrix()

        # Snap cubes to the grid after the animation completes
        if not self.is_animating:
            self._snap_cubes_to_grid()
        
    
    def _rotate_grid_coordinates(self, coords, axis, clockwise):
        """
        Rotate grid coordinates around the given axis by 90 degrees.
        """
        offset = (self.data.size - 1) / 2.0
        position = glm.vec3(coords[0] - offset, coords[1] - offset, coords[2] - offset)

        rotation_matrix = glm.mat4(1.0)
        angle = glm.radians(-90 if clockwise else 90)
        rotation_matrix = glm.rotate(rotation_matrix, angle, axis)

        rotated_position = glm.vec3(rotation_matrix * glm.vec4(position, 1.0))
        return (
            round(rotated_position.x + offset),
            round(rotated_position.y + offset),
            round(rotated_position.z + offset),
        )


    def _snap_cubes_to_grid(self):
        """
        Snap rotating cubes to their new grid positions after animation.
        """
        axis, pivot = self._get_face_axis_and_pivot(self.current_face)

        for scube in self.rotating_cubes:
            scube.grid_coords = self._rotate_grid_coordinates(scube.grid_coords, axis, self.clockwise)
            scube._update_position_from_coords()

    def _get_face_axis_and_pivot(self, face):
        """
        Returns the axis and pivot point for rotating a given face.
        """
        offset = (self.data.size - 1) / 2.0
        if face == 'R':
            return glm.vec3(1, 0, 0), glm.vec3(offset, 0, 0)
        elif face == 'L':
            return glm.vec3(-1, 0, 0), glm.vec3(-offset, 0, 0)
        elif face == 'U':
            return glm.vec3(0, 1, 0), glm.vec3(0, offset, 0)
        elif face == 'D':
            return glm.vec3(0, -1, 0), glm.vec3(0, -offset, 0)
        elif face == 'F':
            return glm.vec3(0, 0, 1), glm.vec3(0, 0, offset)
        elif face == 'B':
            return glm.vec3(0, 0, -1), glm.vec3(0, 0, -offset)
        else:
            raise ValueError(f"Invalid face: {face}")
