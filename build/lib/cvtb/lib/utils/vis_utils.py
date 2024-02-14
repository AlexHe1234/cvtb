import cv2
import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate


class Canvas(app.Canvas):

    vertex_shader = """
    attribute vec3 position;
    attribute vec3 color_in;
    attribute float radius;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    
    varying vec3 color;

    void main() {
        gl_Position = projection * view * model * vec4(position, 1.0);
        gl_PointSize = radius;
        color = color_in;
    }
    """

    fragment_shader = """
    varying vec3 color;

    void main() {
        gl_FragColor = vec4(color, 1.0);
    }
    """
    
    def __init__(self, 
                 point_clouds,  # (F, N, 3) 
                 color=None,  # (N, 3) or (F, N, 3)
                 fps=24,
                 point_size=5,  # float or (N,) or (F, N)
                 ):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600),
                            title='Interactive Point Clouds')
        self.program = gloo.Program(self.vertex_shader, self.fragment_shader)
        self.point_clouds = point_clouds  - np.mean(point_clouds.reshape(-1, 3), axis=0)

        max_value = np.max(np.abs(self.point_clouds))
        self.point_clouds *= 2. / max_value

        self.current_frame = 0
        sequence_speed = 1. / fps
        self.timer = app.Timer(interval=sequence_speed, connect=self.on_timer, start=True)

        # Camera parameters
        self.view = translate((0, 0, -5))
        self.model = np.eye(4, dtype=np.float32)
        self.projection = perspective(45.0, self.size[0] / float(self.size[1]), 1.0, 100.0)

        self.program['model'] = self.model
        self.program['view'] = self.view
        self.program['projection'] = self.projection
        self.point_size = point_size

        self.theta, self.phi = 0, 0
        self.mouse_pos = 0, 0
        self.wheel_pos = 0
        
        self.color = color
        if not self.color is None:
            self.color_seq = len(self.color.shape) == 3

        self.init = True

    def get_point_size(self):
        # return N floats
        num_points = len(self.point_clouds[self.current_frame])
        if isinstance(self.point_size, float) or isinstance(self.point_size, int):
            return np.ones(num_points) * self.point_size
        if not isinstance(self.point_size, np.ndarray):
            raise TypeError(f'Point sizes have type {type(self.point_size)} which is not supported')
        if len(self.point_size.shape) == 1:
            return self.point_size
        if len(self.point_size.shape) == 2:
            return self.point_size[self.current_frame]
        raise ValueError(f'Point sizes array have shape {self.point_size.shape} which is not supported (and weird also)')
    
    def get_point_color(self):
        if self.color is not None:
            if not self.color_seq:
                return self.color
            else:
                return self.color[self.current_frame]
        else:
            return np.ones_like(self.point_clouds[self.current_frame])

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        current_point_cloud = self.point_clouds[self.current_frame]
        self.program['position'] = current_point_cloud.astype(np.float32)
        self.program['radius'] = self.get_point_size().astype(np.float32)
        self.program['color_in'] = self.get_point_color().astype(np.float32)
        self.program.draw('points')

    def on_resize(self, event):
        if not hasattr(self, 'init'): return
        self.projection = perspective(45.0, event.size[0] / float(event.size[1]), 1.0, 100.0)
        self.program['projection'] = self.projection

    def on_mouse_move(self, event):
        x, y = event.pos
        dx, dy = x - self.mouse_pos[0], y - self.mouse_pos[1]
        self.mouse_pos = (x, y)

        if event.is_dragging:
            self.theta += dx
            self.phi += dy

            self.model = np.dot(rotate(self.theta, (0, 1, 0)), rotate(self.phi, (1, 0, 0)))
            self.program['model'] = self.model
            self.update()

    def on_mouse_wheel(self, event):
        self.wheel_pos += event.delta[1]
        self.view = translate((0, 0, -5 - 0.1 * self.wheel_pos))
        self.program['view'] = self.view
        self.update()

    def on_timer(self, event):
        self.current_frame += 1
        self.current_frame %= len(self.point_clouds)
        self.update()


def hsv_to_rgb_opencv(hsv_array):
    # Ensure that the input array has the correct shape (N, 3)
    assert hsv_array.shape[1] == 3, "Input array must have shape (N, 3)"
    # Convert HSV array to RGB array using OpenCV
    hsv_array_uint8 = (hsv_array * 255).astype(np.uint8)  # Convert to uint8
    rgb_array_uint8 = cv2.cvtColor(hsv_array_uint8[None], cv2.COLOR_HSV2RGB)[0]
    # Normalize back to [0, 1]
    rgb_array = rgb_array_uint8 / 255.0
    return rgb_array


# Map the 0-1 value to the Hue component in HSV
def generate_gradient_color(value: np.ndarray,
                            start=0.4,
                            end=0.65,
                            ):
    N = value.shape[0]
    value = (value - value.min()) / (value.max() - value.min())
    value = start + (end - start) * value
    hsv = np.ones((N, 3), dtype=np.float32)
    hsv[:, 0] *= value    
    rgb = hsv_to_rgb_opencv(hsv)

    return rgb


def generate_gradient_color_from_coords(pcd: np.ndarray, 
                                        start=0.4, 
                                        end=0.65, 
                                        use_index: bool = False, 
                                        use_z_axis: bool = True):
    # use_index would simply use index for coloring while False analysis point direction and use that
    if len(pcd.shape) == 2:
        N = pcd.shape[0]
    elif len(pcd.shape) == 3:
        N = pcd.shape[1]
        pcd = pcd[0]
    else:
        raise ValueError(f'Why does point cloud has shape {pcd.shape}')
    
    if use_index:
        return generate_gradient_color(np.arange(N), start=start, end=end)
    else:
        if not use_z_axis:
            # get main direction est.
            center = pcd.mean(axis=0)
            print('Finding main direction, this could be time consuming')
            *_, v = np.linalg.svd(pcd - center)
            vec = v[0]
        else:
            vec = np.array([0., 0., 1.])
        align_value = pcd @ vec.T
        align_ordering = np.argsort(np.argsort(align_value))
        return generate_gradient_color(align_ordering.astype(np.float32), start=start, end=end)
    

def generate_synthetic_point_cloud(num_points):
    return np.random.rand(num_points, 3) * 2 - 1
