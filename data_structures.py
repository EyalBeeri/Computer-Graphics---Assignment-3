import glm
import math

class SmallCube:
    """
    Represents one small sub-cube in the Rubik's cube.
    Stores index, position, orientation, and a 4x4 model matrix.
    """
    def __init__(self, index, position=glm.vec3(0), rotation=glm.vec3(0)):
        self.index = index
        self.position = glm.vec3(position)
        # Euler angles for orientation (in degrees)
        self.rotation = glm.vec3(rotation)
        self.model_matrix = glm.mat4(1.0)
        self.update_model_matrix()

    def update_model_matrix(self):
        """
        For the standard Rubik's assignment, we typically do:
        model = rot * trans * scale
        (rot about global center)
        But you can adapt to your preference.
        """
        # Scale
        scl = glm.scale(glm.mat4(1.0), glm.vec3(1.0))

        # Rotation (apply Z, Y, X in that order, or any order you prefer)
        # We'll store rotation in degrees but convert to radians for glm.rotate
        rot_z = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation.z), glm.vec3(0, 0, 1))
        rot_y = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation.y), glm.vec3(0, 1, 0))
        rot_x = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation.x), glm.vec3(1, 0, 0))
        rot = rot_z * rot_y * rot_x

        # Translation
        trans = glm.translate(glm.mat4(1.0), self.position)

        # For rotations around global center:
        #   model = rot * trans * scl
        self.model_matrix = rot * trans * scl


class RubiksData:
    """
    Holds the sub-cubes for a NxNxN Rubik's cube, with N=2..5 or more.
    By default, N=3 => 27 sub-cubes (or 26 if you skip the internal hidden center).
    """
    def __init__(self, size=3):
        self.size = size
        self.sub_cubes = []
        self._build_cubes()

    def _build_cubes(self):
        """
        Build the NxNxN sub-cubes, each offset from the center so that (0,0,0) is the global center.
        If size=3, we have indices from 0..26 (27 cubes), or skip the internal if you want exactly 26 visible.
        """
        offset = (self.size - 1) / 2.0
        index = 0
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    # Optionally skip the very center if size=3:
                    # if self.size == 3 and x == 1 and y == 1 and z == 1:
                    #    continue
                    pos = glm.vec3(x - offset, y - offset, z - offset)
                    cube = SmallCube(index, position=pos)
                    self.sub_cubes.append(cube)
                    index += 1
