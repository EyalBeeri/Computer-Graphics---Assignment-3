import numpy as np
import glm
from math import radians

class CubePiece:
    def __init__(self, position, index):
        self.initial_position = position.copy()
        self.current_position = position.copy()
        self.index = index
        self.rotation = [0, 0, 0]  # rotation angles around x, y, z axes
        self.faces = self._determine_faces()

    def _determine_faces(self):
        """Determine which faces this piece belongs to based on its position"""
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
        """Recalculate face membership based on current position"""
        self.faces = []
        x, y, z = self.current_position
        if abs(x - 1.1) < 0.1: self.faces.append('R')
        if abs(x + 1.1) < 0.1: self.faces.append('L')
        if abs(y - 1.1) < 0.1: self.faces.append('U')
        if abs(y + 1.1) < 0.1: self.faces.append('D')
        if abs(z - 1.1) < 0.1: self.faces.append('F')
        if abs(z + 1.1) < 0.1: self.faces.append('B')

class RubiksCubeState:
    def __init__(self):
        self.pieces = {}
        self.current_rotation = None
        self.rotation_angle = 90
        self.rotation_direction = 1
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
            'R': ('x', 1),  # Right face rotates around x-axis
            'L': ('x', -1),  # Left face rotates around x-axis
            'U': ('y', 1),  # Up face rotates around y-axis
            'D': ('y', -1),  # Down face rotates around y-axis
            'F': ('z', 1),  # Front face rotates around z-axis
            'B': ('z', -1)  # Back face rotates around z-axis
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
        print(f"{prefix}Piece {piece_index}:")
        print(f"  Position: [{piece.current_position[0]:.2f}, {piece.current_position[1]:.2f}, {piece.current_position[2]:.2f}]")
        print(f"  Rotation: [{piece.rotation[0]:.2f}, {piece.rotation[1]:.2f}, {piece.rotation[2]:.2f}]")
        print(f"  Faces: {piece.faces}")

    def debug_print_face(self, face):
        """Print all pieces in a face"""
        pieces = self.get_face_pieces(face)
        print(f"\nFace {face} contains {len(pieces)} pieces:")
        for piece_index in pieces:
            self.debug_print_piece(piece_index, "  ")

    def rotate_face(self, face):
        """Execute a 90-degree clockwise rotation of the specified face"""
        print(f"\n=== Starting rotation of face {face} ===")
        print("Before rotation:")
        self.debug_print_face(face)

        pieces_to_rotate = self.get_face_pieces(face)

        # Set rotation direction
        direction = -1 if face in ['L', 'D', 'B'] else 1
        angle = 90 * direction

        # Create rotation matrix using glm
        rotation_matrix = glm.mat4(1.0)
        axis = ""
        if face in ['R', 'L']:
            rotation_matrix = glm.rotate(rotation_matrix, glm.radians(angle), glm.vec3(1, 0, 0))
            axis = "X"
        elif face in ['U', 'D']:
            rotation_matrix = glm.rotate(rotation_matrix, glm.radians(angle), glm.vec3(0, 1, 0))
            axis = "Y"
        elif face in ['F', 'B']:
            rotation_matrix = glm.rotate(rotation_matrix, glm.radians(angle), glm.vec3(0, 0, 1))
            axis = "Z"

        print(f"\nApplying {angle}° rotation around {axis} axis")

        # Apply rotation to each piece
        for piece_index in pieces_to_rotate:
            piece = self.state.pieces[piece_index]
            print(f"\nRotating piece {piece_index}:")
            print(f"  Before: pos={[f'{x:.2f}' for x in piece.current_position]}, "
                  f"rot={[f'{x:.2f}' for x in piece.rotation]}")

            # Convert position to vec4 for matrix multiplication
            pos = glm.vec4(piece.current_position[0], piece.current_position[1],
                           piece.current_position[2], 1.0)

            # Apply rotation
            rotated_pos = rotation_matrix * pos
            piece.current_position = [rotated_pos.x, rotated_pos.y, rotated_pos.z]

            # Update piece rotation
            if face in ['R', 'L']:
                piece.rotation[0] = (piece.rotation[0] + angle) % 360
            elif face in ['U', 'D']:
                piece.rotation[1] = (piece.rotation[1] + angle) % 360
            elif face in ['F', 'B']:
                piece.rotation[2] = (piece.rotation[2] + angle) % 360

            print(f"  After:  pos={[f'{x:.2f}' for x in piece.current_position]}, "
                  f"rot={[f'{x:.2f}' for x in piece.rotation]}")

        print("\nAfter rotation:")
        self.debug_print_face(face)
        print("=== Rotation complete ===\n")

        for piece_index in pieces_to_rotate:
            self.state.pieces[piece_index].update_faces()

        return True

    def get_face_pieces(self, face):
        """Get all pieces that belong to the specified face"""
        coords = {
            'R': (0, 1.1),
            'L': (0, -1.1),
            'U': (1, 1.1),
            'D': (1, -1.1),
            'F': (2, 1.1),
            'B': (2, -1.1)
        }

        axis_index, value = coords[face]
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
        face_keys = {
            'R': 'R',
            'L': 'L',
            'U': 'U',
            'D': 'D',
            'F': 'F',
            'B': 'B'
        }

        if key in face_keys:
            print(f"\n=== Processing keyboard input: {key} ===")
            result = self.rotate_face(face_keys[key])
            print(f"=== Keyboard processing complete: {key} ===\n")
            return result
        return False

