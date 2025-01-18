import math
import numpy as np
import glm
from const import STEP_SIZE

class CubePiece:
    def __init__(self, position, index, N=3):
        self.initial_position = position.copy()
        self.index = index
        self.N = N  # Store N so we know how big the cube is
        self.transform = glm.translate(glm.mat4(1.0), glm.vec3(*position))
        self.faces = self._determine_faces()

    def apply_rotation(self, rotation_matrix):
        # Compose the new rotation with existing transform
        self.transform = rotation_matrix * self.transform

    @property
    def current_position(self):
        """Extract position from transformation matrix"""
        return [
            self.transform[3][0],
            self.transform[3][1],
            self.transform[3][2]
        ]

    def get_orientation(self):
        """Extract Euler angles (in radians) from the transformation matrix in XYZ order."""
        rotation_matrix = glm.mat3(self.transform)  # Extract the 3x3 rotation matrix
        sy = math.sqrt(rotation_matrix[0][0] ** 2 + rotation_matrix[1][0] ** 2)
        singular = sy < 1e-6  # Check for gimbal lock

        if not singular:
            x = math.atan2(rotation_matrix[2][1], rotation_matrix[2][2])
            y = math.atan2(-rotation_matrix[2][0], sy)
            z = math.atan2(rotation_matrix[1][0], rotation_matrix[0][0])
        else:
            # Handle gimbal lock: only two angles are independent
            x = math.atan2(-rotation_matrix[1][2], rotation_matrix[1][1])
            y = math.atan2(-rotation_matrix[2][0], sy)
            z = 0  # Arbitrarily set to zero

        return x, y, z  # Returns Euler angles in radians

    def _determine_faces(self):
        """
        For an N×N cube, compute which outer faces this piece belongs to.
        For example, for a 3×3 cube, the outer layers have coordinates ±(step_size).
        For a 4×4 cube, ±(1.5*step_size), etc.
        """
        faces = []
        # The offset is how many steps you go from 0 to reach the outermost layer.
        # For N=3, offset=1; for N=4, offset=1.5; for N=5, offset=2, etc.
        offset = (self.N - 1) / 2.0

        # The step is how far apart each piece is placed. You've been using step_size.
        step = STEP_SIZE

        # Threshold to decide if a coordinate is "close enough" to the outer layer
        threshold = 0.05

        # Current piece position in (x, y, z)
        x, y, z = self.initial_position

        # Compare each coordinate with ±(offset * step)
        # Right
        if abs(x - ( offset * step )) < threshold:
            faces.append('R')
        # Left
        if abs(x - ( -offset * step )) < threshold:
            faces.append('L')
        # Up
        if abs(y - ( offset * step )) < threshold:
            faces.append('U')
        # Down
        if abs(y - ( -offset * step )) < threshold:
            faces.append('D')
        # Front
        if abs(z - ( offset * step )) < threshold:
            faces.append('F')
        # Back
        if abs(z - ( -offset * step )) < threshold:
            faces.append('B')

        return faces

    def update_faces(self):
        """Recalculate face membership based on transformed position and orientation"""
        self.faces = []
        # Extract position from transform
        pos = self.current_position

        # Get the transformed basis vectors
        right = glm.normalize(glm.vec3(self.transform[0][0], self.transform[0][1], self.transform[0][2]))
        up = glm.normalize(glm.vec3(self.transform[1][0], self.transform[1][1], self.transform[1][2]))
        forward = glm.normalize(glm.vec3(self.transform[2][0], self.transform[2][1], self.transform[2][2]))

        # Check face alignment using dot products
        if abs(pos[0] - STEP_SIZE) < 0.1: self.faces.append('R')
        if abs(pos[0] + STEP_SIZE) < 0.1: self.faces.append('L')
        if abs(pos[1] - STEP_SIZE) < 0.1: self.faces.append('U')
        if abs(pos[1] + STEP_SIZE) < 0.1: self.faces.append('D')
        if abs(pos[2] - STEP_SIZE) < 0.1: self.faces.append('F')
        if abs(pos[2] + STEP_SIZE) < 0.1: self.faces.append('B')

class RubiksCubeState:
    def __init__(self, N=3):
        self.N = N
        self.pieces = {}
        self._initialize_pieces()

    def _initialize_pieces(self):
        """
        For an NxN cube, we'll create N^3 pieces, each offset by (x - offset),
        (y - offset), (z - offset) in steps of step_size (or whatever your gap is).
        """
        index = 0
        offset = (self.N - 1) / 2.0  # For 3x3, offset=1; for 4x4, offset=1.5, etc.

        for x in range(self.N):
            for y in range(self.N):
                for z in range(self.N):
                    # Compute actual position in 3D
                    px = (x - offset) * STEP_SIZE
                    py = (y - offset) * STEP_SIZE
                    pz = (z - offset) * STEP_SIZE
                    position = [px, py, pz]

                    # Create a piece
                    self.pieces[index] = CubePiece(position, index)
                    index += 1

    def get_face_pieces(self, face):
        axis_index, value = self.face_coordinates[face]
        face_pieces = []
        for idx, piece in self.state.pieces.items():
            pos_value = piece.current_position[axis_index]
            if abs(pos_value - value) < 0.1:
                face_pieces.append(idx)
        return face_pieces


    def update_piece_position(self, index, new_position):
        """Update the position of a specific piece"""
        self.pieces[index].current_position = new_position

    def update_piece_rotation(self, index, rotation_delta):
        """Update the rotation of a specific piece"""
        piece = self.pieces[index]
        piece.rotation = [
            (piece.rotation[0] + rotation_delta[0]) % 360,
            (piece.rotation[1] + rotation_delta[1]) % 360,
            (piece.rotation[2] + rotation_delta[2]) % 360
        ]


class RubiksCubeController:
    def __init__(self, N=3):
        self.state = RubiksCubeState(N=N)
        self.N = N
        self.offset = (N - 1) / 2.0
        self.direction = 1
        self.angle = 90
        self.half_rotated_faces = []
        
        # Add center shift tracking
        self.center_shift = {'x': 0, 'y': 0, 'z': 0}  # Tracks shifts on each axis
        
        # Update face rotations with axis information
        self.face_rotations = {
            'R': glm.vec3(1, 0, 0),
            'L': glm.vec3(1, 0, 0),
            'U': glm.vec3(0, 1, 0),
            'D': glm.vec3(0, 1, 0),
            'F': glm.vec3(0, 0, 1),
            'B': glm.vec3(0, 0, 1),
            'A': 180,
            'Z': 45
        }

        # Base coordinates for faces (will be adjusted based on center shift)
        self.base_face_coordinates = {
            'R': (0, +self.offset*STEP_SIZE),  # x = +offset*step_size
            'L': (0, -self.offset*STEP_SIZE),  # x = -offset*step_size
            'U': (1, +self.offset*STEP_SIZE),  # y = +offset*step_size
            'D': (1, -self.offset*STEP_SIZE),  # y = -offset*step_size
            'F': (2, +self.offset*STEP_SIZE),  # z = +offset*step_size
            'B': (2, -self.offset*STEP_SIZE),  # z = -offset*step_size
        }

        self.is_animating = False
        self.animation_progress = 0.0
        self.animation_speed = 0.05
        self.current_rotation_face = None
        self.pieces_to_animate = []
        self.rotation_axis = None
        self.target_angle = 0
        self.rotation_center = None
        self.animation_direction = 1
        # Add storage for initial transforms
        self.initial_transforms = {}
        
        # Initialize face_coordinates based on current center
        self.update_face_coordinates()

    def start_face_rotation(self, face):
        """Initialize the rotation animation for a face"""
        if self.is_animating:
            return False

        if not self.can_rotate_face(face):
            print("Cannot rotate face due to blocking")
            return False

        self.pieces_to_animate = self.get_face_pieces(face)
        self.current_rotation_face = face
        self.is_animating = True
        self.animation_progress = 0.0
        self.animation_direction = self.direction

        # Store initial transforms by creating new matrices
        self.initial_transforms = {
            piece_idx: glm.mat4(glm.mat4x4(self.state.pieces[piece_idx].transform))
            for piece_idx in self.pieces_to_animate
        }

        # Set rotation axis
        self.rotation_axis = self.face_rotations[face]

        # Set target angle based on current angle setting
        self.target_angle = self.angle

        # Calculate rotation center
        self.rotation_center = glm.vec3(0.0)
        if face in ['R', 'L']:
            self.rotation_center.x = self.face_coordinates[face][1]
        elif face in ['U', 'D']:
            self.rotation_center.y = self.face_coordinates[face][1]
        elif face in ['F', 'B']:
            self.rotation_center.z = self.face_coordinates[face][1]

        print(f"Starting rotation of {face} face")
        print(f"Target angle: {self.target_angle}")
        print(f"Direction: {self.animation_direction}")

        return True

    def update_animation(self):
        """Update the animation progress and apply rotation"""
        if not self.is_animating:
            return

        # Increment animation progress
        self.animation_progress += self.animation_speed

        if self.animation_progress >= 1.0:
            self._finish_animation()
            return

        # Calculate current angle for this frame
        current_angle = self.target_angle * self.animation_progress * -self.animation_direction

        # Create the rotation transformation
        rotation_mat = glm.mat4(1.0)
        # First translate to origin
        rotation_mat = glm.translate(rotation_mat, -self.rotation_center)
        # Apply rotation
        rotation_mat = glm.rotate(rotation_mat, glm.radians(current_angle), self.rotation_axis)
        # Translate back
        rotation_mat = glm.translate(rotation_mat, self.rotation_center)

        # Apply the current rotation to each piece
        for piece_index in self.pieces_to_animate:
            piece = self.state.pieces[piece_index]
            # Start from the initial transform and apply new rotation
            piece.transform = rotation_mat * self.initial_transforms[piece_index]

    def _finish_animation(self):
        """Clean up after animation is complete"""
        # Ensure final rotation is exactly at target angle
        final_angle = self.target_angle * -self.animation_direction

        # Create final rotation matrix
        final_rotation = glm.mat4(1.0)
        final_rotation = glm.translate(final_rotation, -self.rotation_center)
        final_rotation = glm.rotate(final_rotation, glm.radians(final_angle), self.rotation_axis)
        final_rotation = glm.translate(final_rotation, self.rotation_center)

        # Apply final rotation to pieces
        for piece_index in self.pieces_to_animate:
            piece = self.state.pieces[piece_index]
            piece.transform = final_rotation * self.initial_transforms[piece_index]

        self.is_animating = False
        self.animation_progress = 0.0

        # Update faces for rotated pieces
        for piece_index in self.pieces_to_animate:
            self.state.pieces[piece_index].update_faces()

        # Handle half-rotation state
        if self.angle == 45:
            if self.current_rotation_face in self.half_rotated_faces:
                self.half_rotated_faces.remove(self.current_rotation_face)
            else:
                self.half_rotated_faces.append(self.current_rotation_face)

        # Reset animation state
        self.pieces_to_animate = []
        self.current_rotation_face = None
        self.rotation_axis = None
        self.rotation_center = None
        self.initial_transforms.clear()

    def shift_center(self, axis, direction):
        """
        Shift the center of rotation along specified axis.
        axis: 'x', 'y', or 'z'
        direction: 1 or -1
        """
        # Check if the new position would be within bounds
        new_shift = self.center_shift[axis] + direction
        if -self.offset <= new_shift <= self.offset:
            self.center_shift[axis] = new_shift
            self.update_face_coordinates()
            print(f"Center shifted on {axis} axis to {self.center_shift[axis]}")
            return True
        return False
    
    def update_face_coordinates(self):
        """Update face coordinates based on current center shift"""
        step = STEP_SIZE  # The spacing between pieces
        
        # Create new face coordinates dictionary
        self.face_coordinates = {}
        
        # Adjust each face's position based on center shift
        for face, (axis, value) in self.base_face_coordinates.items():
            new_value = value
            
            # Adjust the coordinate based on center shift
            if axis == 0:  # X-axis faces (R/L)
                new_value = value - (self.center_shift['x'] * step)
            elif axis == 1:  # Y-axis faces (U/D)
                new_value = value - (self.center_shift['y'] * step)
            elif axis == 2:  # Z-axis faces (F/B)
                new_value = value - (self.center_shift['z'] * step)
                
            self.face_coordinates[face] = (axis, new_value)

    def toggle_direction(self):
        if self.direction == 1:
            self.direction = -1
        else:
            self.direction = 1
    def debug_print_piece(self, piece_index, prefix=""):
        """Helper method to print detailed piece information"""
        piece = self.state.pieces[piece_index]
        orientation = piece.get_orientation()
        print(f"{prefix}Piece {piece_index}:")
        print(f"  Position: [{piece.current_position[0]:.2f}, {piece.current_position[1]:.2f}, {piece.current_position[2]:.2f}]")
        print(f"  Rotation: [{glm.degrees(orientation[0]):.2f}, {glm.degrees(orientation[1]):.2f}, {glm.degrees(orientation[2]):.2f}]")
        print(f"  Faces: {piece.faces}")

    def debug_print_face(self, face):
        """Print all pieces in a face"""
        pieces = self.get_face_pieces(face)
        print(f"\nFace {face} contains {len(pieces)} pieces:")
        for piece_index in pieces:
            self.debug_print_piece(piece_index, "  ")

    def can_rotate_face(self, face):
        """Check if the given face can be rotated based on current half-rotated state."""
        if not self.half_rotated_faces:
            return True  # No restrictions if no face is half-rotated

        # If the requested face is already half-rotated, allow rotation
        if face in self.half_rotated_faces:
            return True

        # Only allow rotation of directly related faces
        if 'R' in self.half_rotated_faces and face == 'L':
            return True
        if 'L' in self.half_rotated_faces and face == 'R':
            return True
        if 'U' in self.half_rotated_faces and face == 'D':
            return True
        if 'D' in self.half_rotated_faces and face == 'U':
            return True
        if 'F' in self.half_rotated_faces and face == 'B':
            return True
        if 'B' in self.half_rotated_faces and face == 'F':
            return True

        return False  # Block rotation for other faces

    def rotate_face(self, face):
        """Modified to use animation system"""
        return self.start_face_rotation(face)

    def get_face_pieces(self, face):
        """Get all pieces that belong to the specified face"""
        axis_index, value = self.face_coordinates[face]
        pieces = []

        print(f"\nFinding pieces for face {face}:")
        print(f"Checking coordinate {axis_index} for value {value}")

        for index, piece in self.state.pieces.items():
            pos_value = piece.current_position[axis_index]
            if abs(pos_value - value) < 0.1:
                print(f"Found piece {index}: coordinate {axis_index} = {pos_value:.2f}")
                pieces.append(index)
            else:
                print(f"Skipped piece {index}: coordinate {axis_index} = {pos_value:.2f}")

        return pieces

    def _create_rotation_matrix(self, axis, angle):
        """Create a 3D rotation matrix for the given axis and angle"""
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        t = 1 - c

        return np.array([
            [t * x * x + c, t * x * y - z * s, t * x * z + y * s],
            [t * x * y + z * s, t * y * y + c, t * y * z - x * s],
            [t * x * z - y * s, t * y * z + x * s, t * z * z + c]
        ])

    def process_keyboard(self, key):
        """Process keyboard input for face rotations and center shifts"""
        face_keys = {'R': 'R', 'L': 'L', 'U': 'U', 'D': 'D', 'F': 'F', 'B': 'B'}
        
        # Handle center shift controls
        if key == 'RIGHT':
            return self.shift_center('x', -1)
        elif key == 'LEFT':
            return self.shift_center('x', 1)
        elif key == 'UP':
            return self.shift_center('y', -1)
        elif key == 'DOWN':
            return self.shift_center('y', 1)
        elif key == 'I':
            return self.shift_center('z', -1)
        elif key == 'O':
            return self.shift_center('z', 1)
        # Handle rotation direction toggle
        elif key == 'SPACE':
            self.toggle_direction()
        # Handle rotation angle changes
        elif key == 'A':
            if self.angle == 45:
                self.angle = 90
            elif self.angle == 90:
                self.angle = 180
            elif self.angle == 180:
                return
        elif key == 'Z':
            if self.angle == 45:
                return
            elif self.angle == 90:
                self.angle = 45
            elif self.angle == 180:
                self.angle = 90
        # Handle face rotations
        elif key in face_keys:
            print(f"\n=== Processing keyboard input: {key} ===")
            self.last_face = face_keys[key]
            result = self.rotate_face(face_keys[key])
            print(f"=== Keyboard processing complete: {key} ===\n")
            return result

        return False

