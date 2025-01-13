import glm
import random

class SubCube:
    def __init__(self, idx, x, y, z):
        self.idx = idx
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z

        # Store orientation separately from translation:
        self.orientation = glm.mat4(1.0)
        
        self.model_matrix = glm.mat4(1.0)
        self._update_model_matrix()

    def _update_model_matrix(self):
        # Combine translation * orientation
        translate_mat = glm.translate(glm.mat4(1.0),
                                      glm.vec3(self.grid_x, self.grid_y, self.grid_z))
        self.model_matrix = translate_mat * self.orientation

    def update_grid_pos(self, x, y, z):
        # Update integer grid position after a 90° face turn
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z
        # Recompute with the same orientation
        self._update_model_matrix()



class RubiksData:
    """
    This class holds all the Rubik's sub-cube data and any associated logic
    for indexing, random mixers, or helper methods to update the geometry data.
    """
    def __init__(self, size=3):
        """
        :param size: the Rubik's size, typically 3 for a classic 3x3.
        """
        self.size = size
        self.sub_cubes = []
        self.solver = None  # Will hold a RubikSolver instance

        self._build_sub_cubes()

    def _build_sub_cubes(self):
        """
        Build the sub-cubes for a NxNxN Rubik's cube. 
        We store them in 'sub_cubes'. 
        For size=3, we have 3^3 = 27 sub-cubes (including the hidden center if you want).
        Typically, a 3x3 visible has 26, but often we still track the 27th internal piece 
        in code. Adapt as you wish.
        We place them in [-1, 0, 1] in x, y, z for a standard 3x3.
        """
        # We'll use a simple index to color pick each sub-cube uniquely:
        index_counter = 0

        # Example: For size=3 => range(-1, 2) => -1, 0, 1
        offset = self.size // 2  # 3//2=1, 4//2=2, ...
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    gx = x - offset
                    gy = y - offset
                    gz = z - offset
                    sub = SubCube(index_counter, gx, gy, gz)
                    self.sub_cubes.append(sub)
                    index_counter += 1

    def get_sub_cube_by_index(self, idx):
        """
        Return the sub-cube with a given picking index.
        """
        for sc in self.sub_cubes:
            if sc.idx == idx:
                return sc
        return None

    def random_mixer(self, rubiks_renderer, steps=10):
        """
        Randomly rotate faces to scramble the cube.
        This calls 'rubiks_renderer.rotate_face(...)' with random faces 
        and random directions (clockwise / not).
        """
        faces = ['R', 'L', 'U', 'D', 'F', 'B']
        for _ in range(steps):
            face = random.choice(faces)
            clockwise = random.choice([True, False])
            rubiks_renderer.rotate_face(face, clockwise)
