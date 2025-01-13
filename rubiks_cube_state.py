import math

import numpy as np
import glm
from math import radians

import pyglm


class CubePiece:
    def __init__(self, position, index):
        self.initial_position = position.copy()
        self.index = index
        # Initialize transformation matrix
        self.transform = glm.mat4(1.0)
        self.transform = glm.translate(self.transform, glm.vec3(*position))
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
        """Determine which faces this piece belongs to based on its initial position"""
        faces = []
        x, y, z = self.initial_position
        if abs(x - 1.1) < 0.1: faces.append('R')
        if abs(x + 1.1) < 0.1: faces.append('L')
        if abs(y - 1.1) < 0.1: faces.append('U')
        if abs(y + 1.1) < 0.1: faces.append('D')
        if abs(z - 1.1) < 0.1: faces.append('F')
        if abs(z + 1.1) < 0.1: faces.append('B')
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
        if abs(pos[0] - 1.1) < 0.1: self.faces.append('R')
        if abs(pos[0] + 1.1) < 0.1: self.faces.append('L')
        if abs(pos[1] - 1.1) < 0.1: self.faces.append('U')
        if abs(pos[1] + 1.1) < 0.1: self.faces.append('D')
        if abs(pos[2] - 1.1) < 0.1: self.faces.append('F')
        if abs(pos[2] + 1.1) < 0.1: self.faces.append('B')

class RubiksCubeState:
    def __init__(self):
        self.pieces = {}
        self._initialize_pieces()

    def _initialize_pieces(self):
        index = 0
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    position = [x * 1.1, y * 1.1, z * 1.1]
                    self.pieces[index] = CubePiece(position, index)
                    index += 1

    def get_face_pieces(self, face):
        """Get indices of pieces belonging to a specific face"""
        face_pieces = []
        for index, piece in self.pieces.items():
            if face in piece.faces:
                face_pieces.append(index)
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
    def __init__(self):
        self.state = RubiksCubeState()
        self.face_rotations = {
            'R': (glm.vec3(1, 0, 0), 1),   # Right face rotates around x-axis
            'L': (glm.vec3(1, 0, 0), -1),  # Left face rotates around x-axis
            'U': (glm.vec3(0, 1, 0), 1),   # Up face rotates around y-axis
            'D': (glm.vec3(0, 1, 0), -1),  # Down face rotates around y-axis
            'F': (glm.vec3(0, 0, 1), 1),   # Front face rotates around z-axis
            'B': (glm.vec3(0, 0, 1), -1)   # Back face rotates around z-axis
        }
        # Define rotation axes for each face
        self.rotation_axes = {
            'R': glm.vec3(1, 0, 0),
            'L': glm.vec3(1, 0, 0),
            'U': glm.vec3(0, 1, 0),
            'D': glm.vec3(0, 1, 0),
            'F': glm.vec3(0, 0, 1),
            'B': glm.vec3(0, 0, 1)
        }
        # Define which coordinate to check for each face
        self.face_coordinates = {
            'R': (0, 1.1),    # x = 1.1
            'L': (0, -1.1),   # x = -1.1
            'U': (1, 1.1),    # y = 1.1
            'D': (1, -1.1),   # y = -1.1
            'F': (2, 1.1),    # z = 1.1
            'B': (2, -1.1)    # z = -1.1
        }

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

    def rotate_face(self, face):
        """Execute a 90-degree rotation of the specified face"""
        print(f"\n=== Starting rotation of face {face} ===")
        print("Before rotation:")
        self.debug_print_face(face)

        # Get pieces to rotate
        pieces_to_rotate = self.get_face_pieces(face)

        # Get rotation axis and direction
        axis, direction = self.face_rotations[face]
        angle = 90 * direction

        # Create rotation matrix
        rotation_center = glm.vec3(0.0)
        if face in ['R', 'L']:
            rotation_center.x = self.face_coordinates[face][1]
        elif face in ['U', 'D']:
            rotation_center.y = self.face_coordinates[face][1]
        elif face in ['F', 'B']:
            rotation_center.z = self.face_coordinates[face][1]

        # Create the rotation transformation
        rotation_mat = glm.mat4(1.0)
        # First translate to origin
        rotation_mat = glm.translate(rotation_mat, -rotation_center)
        # Apply rotation
        rotation_mat = glm.rotate(rotation_mat, glm.radians(angle), axis)
        # Translate back
        rotation_mat = glm.translate(rotation_mat, rotation_center)

        print(f"\nApplying {angle}° rotation around axis {axis.x}, {axis.y}, {axis.z}")

        # Apply rotation to each piece
        for piece_index in pieces_to_rotate:
            piece = self.state.pieces[piece_index]
            print(f"\nRotating piece {piece_index}:")
            print(f"  Before: pos={[f'{x:.2f}' for x in piece.current_position]}")

            # Apply the new rotation
            piece.transform = rotation_mat * piece.transform

            print(f"  After:  pos={[f'{x:.2f}' for x in piece.current_position]}")

        print("\nAfter rotation:")
        self.debug_print_face(face)
        print("=== Rotation complete ===\n")

        # Update faces for rotated pieces
        for piece_index in pieces_to_rotate:
            self.state.pieces[piece_index].update_faces()

        return True

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
        """Process keyboard input for face rotations"""
        face_keys = {'R': 'R', 'L': 'L', 'U': 'U', 'D': 'D', 'F': 'F', 'B': 'B'}

        if key in face_keys:
            print(f"\n=== Processing keyboard input: {key} ===")
            result = self.rotate_face(face_keys[key])
            print(f"=== Keyboard processing complete: {key} ===\n")
            return result
        return False

