import glfw
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
from PIL import Image
import glm
from camera import Camera
from rubiks_cube_state import RubiksCubeController
from const import N
class RubiksCube:
	def __init__(self, width=800, height=600, N=3):
		self.N = N
		self.camera = Camera(width, height)
		self.cube_controller = RubiksCubeController(N=self.N)

		with open('shaders/basic.vert', 'r') as f:
			self.VERTEX_SHADER = f.read()
		with open('shaders/basic.frag', 'r') as f:
			self.FRAGMENT_SHADER = f.read()

		self.COLORS = {
			'white': [1.0, 1.0, 1.0],
			'yellow': [1.0, 1.0, 0.0],
			'red': [1.0, 0.0, 0.0],
			'orange': [1.0, 0.5, 0.0],
			'green': [0.0, 1.0, 0.0],
			'blue': [0.0, 0.0, 1.0]
		}

		# Same vertices as before but with texture coordinates
		self.vertices = np.array([
			# Format: x, y, z, r, g, b, tex_x, tex_y
			# Front face (white)
			-0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 1.0,
			0.5, -0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0,
			0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0,
			-0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0,

			# Back face (yellow)
			-0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 1.0,
			0.5, -0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 1.0,
			0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 1.0, 0.0,
			-0.5, 0.5, -0.5, 1.0, 1.0, 0.0, 0.0, 0.0,

			# Right face (red)
			0.5, -0.5, -0.5, 1.0, 0.0, 0.0, 0.0, 1.0,
			0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 1.0, 1.0,
			0.5, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0,
			0.5, -0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0,

			# Left face (orange)
			-0.5, -0.5, -0.5, 1.0, 0.5, 0.0, 0.0, 1.0,
			-0.5, 0.5, -0.5, 1.0, 0.5, 0.0, 1.0, 1.0,
			-0.5, 0.5, 0.5, 1.0, 0.5, 0.0, 1.0, 0.0,
			-0.5, -0.5, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0,

			# Top face (green)
			-0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 0.0, 1.0,
			0.5, 0.5, -0.5, 0.0, 1.0, 0.0, 1.0, 1.0,
			0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0, 0.0,
			-0.5, 0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0,

			# Bottom face (blue)
			-0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 0.0, 1.0,
			0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0,
			0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0,
			-0.5, -0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0,
		], dtype=np.float32)

		self.indices = np.array([
			0, 1, 2, 2, 3, 0,  # Front
			4, 5, 6, 6, 7, 4,  # Back
			8, 9, 10, 10, 11, 8,  # Right
			12, 13, 14, 14, 15, 12,  # Left
			16, 17, 18, 18, 19, 16,  # Top
			20, 21, 22, 22, 23, 20  # Bottom
		], dtype=np.uint32)

		self.shader = None
		self.vao = None
		self.vbo = None
		self.ebo = None
		self.texture = None
		
		self.picking_mode = False
		self.selected_cube_id = -1
		self.selected_depth = 1.0  # Keep the depth of the selected cube


	def load_texture(self, path):
		self.texture = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.texture)

		# Set texture wrapping parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)

		# Set texture filtering parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

		# Load image
		try:
			image = Image.open(path)
			image = image.transpose(Image.FLIP_TOP_BOTTOM)  # Flip the image
			if image.mode != 'RGB':
				image = image.convert('RGB')
			img_data = image.tobytes()
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.width, image.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)
			glGenerateMipmap(GL_TEXTURE_2D)
			print(f"Texture loaded successfully: {path}")
		except Exception as e:
			print(f"Error loading texture {path}: {e}")

	def init_gl(self):
		# Compile shaders
		vertex_shader = shaders.compileShader(self.VERTEX_SHADER, GL_VERTEX_SHADER)
		fragment_shader = shaders.compileShader(self.FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
		self.shader = shaders.compileProgram(vertex_shader, fragment_shader)

		# Create and bind VAO
		self.vao = glGenVertexArrays(1)
		glBindVertexArray(self.vao)

		# Create and bind VBO
		self.vbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
		glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)

		# Create and bind EBO
		self.ebo = glGenBuffers(1)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)

		# Position attribute
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * 4, None)
		glEnableVertexAttribArray(0)
		# Color attribute
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(3 * 4))
		glEnableVertexAttribArray(1)
		# Texture coordinate attribute
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * 4, ctypes.c_void_p(6 * 4))
		glEnableVertexAttribArray(2)

		# Load texture
		self.load_texture("textures/plane.png")

		# Enable depth testing
		glEnable(GL_DEPTH_TEST)

	def draw(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glUseProgram(self.shader)

		# Set uniforms
		glUniform4f(glGetUniformLocation(self.shader, "u_Color"), 1.0, 1.0, 1.0, 1.0)
		glUniform1i(glGetUniformLocation(self.shader, "u_PickingMode"), 0)
		glUniform1i(glGetUniformLocation(self.shader, "u_Texture"), 0)

		# Activate texture
		glActiveTexture(GL_TEXTURE0)
		glBindTexture(GL_TEXTURE_2D, self.texture)

		view = self.camera.get_view_matrix()
		projection = self.camera.get_perspective_matrix()

		for piece in self.cube_controller.state.pieces.values():
			# Use the piece's transformation matrix directly
			model = piece.transform

			# Calculate MVP matrix
			mvp = projection * view * model

			# Send MVP matrix to shader
			glUniformMatrix4fv(glGetUniformLocation(self.shader, "u_MVP"), 1, GL_FALSE, glm.value_ptr(mvp))

			# Draw the cube piece
			glBindVertexArray(self.vao)
			glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)


def main():
	if not glfw.init():
		return

	window = glfw.create_window(800, 600, "Rubik's Cube", None, None)
	if not window:
		glfw.terminate()
		return

	glfw.make_context_current(window)

	cube = RubiksCube(width=800, height=600, N=N)
	cube.init_gl()

	def mouse_callback(window, xpos, ypos):
		cube.camera.process_mouse_motion(window, xpos, ypos)
		
	def mouse_button_callback(window, button, action, mods):
		if action == glfw.PRESS and cube.picking_mode:
			# 1) Clear color + depth
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			
			# Temporarily enable picking mode in the shader
			glUseProgram(cube.shader)
			glUniform1i(glGetUniformLocation(cube.shader, "u_PickingMode"), 1)

			# 2) Draw each piece with a unique color
			# For example, set color = (index+1)/255 in R channel. 
			# (For bigger cubes, you can do a fancier mapping.)
			for i, piece in enumerate(cube.cube_controller.state.pieces.values()):
				# i goes from 0..N-1, so picking color can be:
				r = (i+1)/255.0
				g = 0.0
				b = 0.0

				# Send picking color
				glUniform3f(glGetUniformLocation(cube.shader, "u_PickingColor"), r, g, b)

				# Compute MVP, send it, bind VAO, draw...
				model = piece.transform
				view = cube.camera.get_view_matrix()
				projection = cube.camera.get_perspective_matrix()
				mvp = projection * view * model

				glUniformMatrix4fv(glGetUniformLocation(cube.shader, "u_MVP"),
								1, GL_FALSE,
								glm.value_ptr(mvp))
				glBindVertexArray(cube.vao)
				glDrawElements(GL_TRIANGLES, len(cube.indices), GL_UNSIGNED_INT, None)

			# Force finish
			glFlush()
			glFinish()

			# 3) Use glReadPixels to read the color at mouse pos
			x, y = glfw.get_cursor_pos(window)
			# Note: The window's origin is top-left or bottom-left?
			# In many cases, OpenGL expects bottom-left, while your window coords are top-left.
			# If so, do: real_y = window_height - y
			real_y = cube.camera.height - int(y) - 1
			pixel_data = glReadPixels(int(x), int(real_y), 1, 1, GL_RGB, GL_FLOAT)

			picked_r = pixel_data[0][0][0]  # because pixel_data is [height][width][channel]
			picked_g = pixel_data[0][0][1]
			picked_b = pixel_data[0][0][2]

			# Convert back to index
			# If we only stored index in R channel:
			selected_id = int(round(picked_r * 255.0)) - 1

			# 4) Now read depth
			depth_data = glReadPixels(int(x), int(real_y), 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT)
			picked_depth = depth_data[0][0]

			# 5) Store them
			cube.selected_cube_id = selected_id if (selected_id >= 0) else -1
			cube.selected_depth   = picked_depth
			print("Picked cube =", cube.selected_cube_id, " depth=", cube.selected_depth)

			# Done picking => turn off picking mode in the shader
			glUniform1i(glGetUniformLocation(cube.shader, "u_PickingMode"), 0)
			
	def cursor_pos_callback(window, xpos, ypos):
		# If no piece selected, do normal camera controls or nothing
		if cube.selected_cube_id < 0:
			# Maybe your existing camera logic
			cube.camera.process_mouse_motion(window, xpos, ypos)
			return

		# If right button is pressed => translate
		if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
			real_y = cube.camera.height - int(ypos) - 1
			ndc_x = (2.0 * xpos) / cube.camera.width - 1.0
			ndc_y = (2.0 * real_y) / cube.camera.height - 1.0
			ndc_z = 2.0 * cube.selected_depth - 1.0

			clip_coords = glm.vec4(ndc_x, ndc_y, ndc_z, 1.0)
			inv_mvp = glm.inverse(cube.camera.get_perspective_matrix() * cube.camera.get_view_matrix())
			world_coords = inv_mvp * clip_coords
			world_coords /= world_coords.w

			if cube.selected_cube_id != 50:
				piece = cube.cube_controller.state.pieces[cube.selected_cube_id]
				piece_center_local = glm.vec3(piece.current_position)
				new_center = glm.vec3(world_coords.x, world_coords.y, world_coords.z)
				translation_delta = new_center - piece_center_local
				piece.transform = glm.translate(piece.transform, translation_delta)

		# If left button is pressed => rotate around local axes
		elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
			if cube.selected_cube_id != 50:
				piece = cube.cube_controller.state.pieces[cube.selected_cube_id]
				
				# Calculate mouse movement
				dx = xpos - cube.camera.last_mouse_x
				dy = ypos - cube.camera.last_mouse_y
				
				# Get camera vectors
				view_matrix = cube.camera.get_view_matrix()
				camera_right = glm.normalize(glm.vec3(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]))
				camera_up = glm.normalize(glm.vec3(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]))
				
				# Create rotation matrices for both X and Y mouse movement
				rotation_x = glm.rotate(glm.mat4(1.0), glm.radians(-dy * 2.0), camera_right)
				rotation_y = glm.rotate(glm.mat4(1.0), glm.radians(-dx * 2.0), camera_up)
				
				# Combine rotations
				total_rotation = rotation_y * rotation_x
				
				# Get piece's center in world space
				piece_center = glm.vec3(piece.current_position)
				
				# Create the complete transformation:
				# 1. Translate to origin
				translate_to_origin = glm.translate(glm.mat4(1.0), -piece_center)
				# 2. Apply rotation
				# 3. Translate back
				translate_back = glm.translate(glm.mat4(1.0), piece_center)
				
				# Combine all transformations
				final_transform = translate_back * total_rotation * translate_to_origin
				
				# Apply the transformation
				piece.transform = final_transform * piece.transform

		# Update last mouse pos
		cube.camera.last_mouse_x = xpos
		cube.camera.last_mouse_y = ypos



	def scroll_callback(window, xoffset, yoffset):
		cube.camera.process_scroll(yoffset)

	def key_callback(window, key, scancode, action, mods):
		if action == glfw.PRESS:
			if key == glfw.KEY_P:
				cube.picking_mode = not cube.picking_mode
				if not cube.picking_mode:
					cube.selected_cube_id = -1
					cube.selected_depth = None
					print("Picking mode disabled: Deselected cube")
				else:
					print("Picking mode enabled")
			# Center shift controls
			elif key == glfw.KEY_LEFT:
				cube.cube_controller.process_keyboard('LEFT')
			elif key == glfw.KEY_RIGHT:
				cube.cube_controller.process_keyboard('RIGHT')
			elif key == glfw.KEY_UP:
				cube.cube_controller.process_keyboard('UP')
			elif key == glfw.KEY_DOWN:
				cube.cube_controller.process_keyboard('DOWN')
			elif key == glfw.KEY_I:
				cube.cube_controller.process_keyboard('I')
			elif key == glfw.KEY_O:
				cube.cube_controller.process_keyboard('O')
			# Existing rotation controls
			elif key == glfw.KEY_R:
				cube.cube_controller.process_keyboard('R')
			elif key == glfw.KEY_L:
				cube.cube_controller.process_keyboard('L')
			elif key == glfw.KEY_U:
				cube.cube_controller.process_keyboard('U')
			elif key == glfw.KEY_D:
				cube.cube_controller.process_keyboard('D')
			elif key == glfw.KEY_F:
				cube.cube_controller.process_keyboard('F')
			elif key == glfw.KEY_B:
				cube.cube_controller.process_keyboard('B')
			elif key == glfw.KEY_SPACE:
				cube.cube_controller.process_keyboard('SPACE')
			elif key == glfw.KEY_A:
				cube.cube_controller.process_keyboard('A')
			elif key == glfw.KEY_Z:
				cube.cube_controller.process_keyboard('Z')

	glfw.set_cursor_pos_callback(window, mouse_callback)
	glfw.set_scroll_callback(window, scroll_callback)
	glfw.set_key_callback(window, key_callback)
	glfw.set_mouse_button_callback(window, mouse_button_callback)
	glfw.set_cursor_pos_callback(window, cursor_pos_callback)

	glClearColor(0.2, 0.3, 0.3, 1.0)

	while not glfw.window_should_close(window):
		cube.cube_controller.update_animation()

		cube.draw()
		glfw.swap_buffers(window)
		glfw.poll_events()


	glfw.terminate()


if __name__ == "__main__":
	main()