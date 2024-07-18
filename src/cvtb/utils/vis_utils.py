import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

from ..geometry import *


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
                 
                 show_directions: bool = False,
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
        self.play = True
        
        # aux flags
        self.use_aux = False
        self.aux_x = []  # these are time invariant, I hope, so N, 3
        self.aux_r = []  # N,
        self.aux_c = []  # N, 3
        
        if show_directions:
            self.add_aux_direction()
        
        assert len(self.aux_x) == len(self.aux_r) and \
            len(self.aux_r) == len(self.aux_c)
    
    def aux(func):
        def wrapper(*args, **kwargs):
            args[0].use_aux = True
            return func(*args, **kwargs) 
        return wrapper
    
    @aux
    def add_aux_direction(self):
        cone_angle=40.
        cone_ratio=0.8
        length_ratio=0.1
        num_stride=10
        num_points=100
        
        x_arrow = generate_arrow(np.array([0., 0., 0.]),
                                 np.array([1., 0., 0.]),
                                 cone_angle=cone_angle,
                                 cone_ratio=cone_ratio,
                                 length_ratio=length_ratio,
                                 num_stride=num_stride,
                                 num_points=num_points)
        y_arrow = generate_arrow(np.array([0., 0., 0.]),
                                 np.array([0., 1., 0.]),
                                 cone_angle=cone_angle,
                                 cone_ratio=cone_ratio,
                                 length_ratio=length_ratio,
                                 num_stride=num_stride,
                                 num_points=num_points)
        z_arrow = generate_arrow(np.array([0., 0., 0.]),
                                 np.array([0., 0., 1.]),
                                 cone_angle=cone_angle,
                                 cone_ratio=cone_ratio,
                                 length_ratio=length_ratio,
                                 num_stride=num_stride,
                                 num_points=num_points)
        
        arrows = np.concatenate([x_arrow, y_arrow, z_arrow], axis=0).astype(np.float32)
        arrows_color = np.concatenate([np.ones_like(x_arrow) * np.array([1., 0.2, 0.2]),
                                       np.ones_like(y_arrow) * np.array([0.2, 1., 0.2]),
                                       np.ones_like(z_arrow) * np.array([0.2, 0.2, 1.])], axis=0).astype(np.float32)
        
        self.aux_x.append(arrows)
        self.aux_r.append(np.ones(arrows.shape[0], dtype=np.float32) * 5.)
        self.aux_c.append(arrows_color)

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
        
    def get_aux_input(self):
        ret_x = []
        ret_r = []
        ret_c = []
        
        for i in range(len(self.aux_x)):
            ret_x.append(self.aux_x[i])
            ret_r.append(self.aux_r[i])
            ret_c.append(self.aux_c[i])
            
        ret_x = np.concatenate(ret_x, axis=0)
        ret_r = np.concatenate(ret_r, axis=0)
        ret_c = np.concatenate(ret_c, axis=0)
        
        return ret_x, ret_r, ret_c

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        current_point_cloud = self.point_clouds[self.current_frame]
        
        x = current_point_cloud.astype(np.float32)
        r = self.get_point_size().astype(np.float32)
        c = self.get_point_color().astype(np.float32)
        
        if self.use_aux:
            x_, r_, c_ = self.get_aux_input()
            x = np.concatenate([x, x_], axis=0)
            r = np.concatenate([r, r_], axis=0)
            c = np.concatenate([c, c_], axis=0)
        
        self.program['position'] = x
        self.program['radius'] = r
        self.program['color_in'] = c
         
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
        
    def on_key_press(self, event):
        if event.key == 'Space':
            self.play = not self.play

    def on_timer(self, event):
        if self.play:
            self.current_frame += 1
            self.current_frame %= len(self.point_clouds)
            self.update()
        else:
            pass
