import random

class RubikSolver:
    """
    Provides naive random mixer and a *very* simplistic "solver" stub. 
    For a real solver, you'd implement a known algorithm (CFOP, Kociemba, etc.). 
    Here we store and replay moves for demonstration.
    """
    def __init__(self):
        self.move_sequence = []
        self.mix_sequence = []

    def random_mixer(self, rubiks_renderer, steps=10):
        """
        Randomly apply face rotations. We'll store the moves in `mix_sequence`.
        """
        possible_moves = ['R', 'L', 'U', 'D', 'F', 'B']
        self.mix_sequence.clear()

        for _ in range(steps):
            face = random.choice(possible_moves)
            # Randomly pick clockwise or counterclockwise
            clockwise = random.choice([True, False])
            rubiks_renderer.rotate_face(face, clockwise)
            self.mix_sequence.append(face + ("'" if not clockwise else ""))  # e.g. R'

        # Write to mixer.txt
        with open("mixer.txt", "w") as f:
            f.write(" ".join(self.mix_sequence))

    def solve(self, rubiks_renderer):
        """
        A naive "solver": just reverse the moves from the mixer. 
        If a real solver is needed, replace with a real algorithm.
        """
        self.move_sequence.clear()

        # Reverse the random mix moves
        for move in reversed(rubiks_renderer.solver.mix_sequence):
            # If move was R', invert to R, etc.
            if move.endswith("'"):
                face = move[0]
                # The inverted move is "face, clockwise" 
                rubiks_renderer.rotate_face(face, True)  # Clockwise
                self.move_sequence.append(face)
            else:
                face = move
                # The inverted move is "face, counterclockwise"
                rubiks_renderer.rotate_face(face, False)
                self.move_sequence.append(face + "'")

        # Write to solver.txt
        with open("solver.txt", "w") as f:
            f.write(" ".join(self.move_sequence))
