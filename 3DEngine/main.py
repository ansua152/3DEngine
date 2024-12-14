import pygame
import numpy as np
import math

# 초기 설정
WIDTH, HEIGHT = 800, 600
FOV = 60
ERROR = 0.5

class Shape:
    center = np.array([0., 0., 0., 1.])
    vertices = np.zeros(shape=(4,))
    triangle = []
    velocity = np.array([0., 0., 0., 0.])
    acceleration = np.array([0., 0., 0., 0.])
    a_velocity = np.array([0., 0., 0.])
    a_acceleration = 0
    inertia = 0
    mass = 0
    bounce = 1
    
    def translation(self, x, y, z):
        matrixT = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

        self.center += np.array([x, y, z, 0.])

        for vertex in range(0, self.vertices.shape[0]):
            self.vertices[vertex] = np.dot(matrixT, self.vertices[vertex])

    def scale(self, x, y, z):
        matrixS = np.array([
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1]
        ])

        for vertex in range(0, self.vertices.shape[0]):
            self.vertices[vertex] = np.dot(matrixS, self.vertices[vertex])

    def rotate(self, yaw, pitch, roll):
        matrixRy = np.array([
            [math.cos(math.radians(yaw)), 0, math.sin(math.radians(yaw)), 0],
            [0, 1, 0, 0],
            [-math.sin(math.radians(yaw)), 0, math.cos(math.radians(yaw)), 0],
            [0, 0, 0, 1]
        ])
        matrixRx = np.array([
            [1, 0, 0, 0],
            [0, math.cos(math.radians(pitch)), -math.sin(math.radians(pitch)), 0],
            [0, math.sin(math.radians(pitch)), math.cos(math.radians(pitch)), 0],
            [0, 0, 0, 1]
        ])
        matrixRz = np.array([
            [math.cos(math.radians(roll)), -math.sin(math.radians(roll)), 0, 0],
            [math.sin(math.radians(roll)), math.cos(math.radians(roll)), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        matrixLR = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])

        matrixR = np.dot(matrixRy, np.dot(matrixRx, matrixRz))

        for vertex in range(0, self.vertices.shape[0]):
            self.vertices[vertex] = np.dot(matrixR, self.vertices[vertex])
            
    def mesh_represent(self):
        v_plane = []
        for tri in self.triangle:
            v1 = self.vertices[tri[0]]
            v2 = self.vertices[tri[1]]
            v3 = self.vertices[tri[2]]

            for s in range(0, 20):
                for t in range(0, 20):
                    if s + t <= 20:
                        v_plane.append(s / 20 * (v1 - v3) + t / 20 * (v2 - v3) + v3)
        return v_plane
    
    
class Cube(Shape):
    def __init__(self):
        self.mass = 1
        self.inertia = 100
        self.vertices = np.array([
            [-1., -1., -1., 1.],
            [1., -1., -1., 1.],
            [1., 1., -1., 1.],
            [-1., 1., -1., 1.],
            [-1., -1., 1., 1.],
            [1., -1., 1., 1.],
            [1., 1., 1., 1.],
            [-1., 1., 1., 1.],
        ])
        self.center = np.array([0., 0., 0., 1.])

    def gravity(self):
        gravity_a = 0.001
        self.velocity += np.array([0, -gravity_a, 0, 0])
        self.translation(self.velocity[0], self.velocity[1], self.velocity[2])


    def detect_collision(self, plane):
        p1, p2, p3 = plane.vertices[:3]
        normal = np.cross(p3[:3] - p1[:3], p2[:3] - p1[:3])
        normal = normal / np.linalg.norm(normal)

        A, B, C = normal
        D = -np.dot(normal, p1[:3])
        for vertex in self.vertices:
            x, y, z = vertex[:3]
            distance = A * x + B * y + C * z + D

            if abs(distance) < ERROR:
                return True, normal, vertex
            
        return False, normal, np.array([0, 0, 0, 1])


    def handle_collision(self, plane):
        
        iscollision, normal, contact_point = self.detect_collision(plane)        

        if iscollision:
            r_1 = contact_point - self.center
            r_1 = np.array([r_1[0], r_1[1], r_1[2]])

            relative_velocity = self.velocity[:3]
            penetration_velocity = relative_velocity.dot(normal)

            if penetration_velocity > 0:
                return

            j = -(1 + 0.3) * penetration_velocity
            j /= 1 / self.mass + 1 / plane.mass

            impulse = -normal * j
            
            self.velocity -= np.array([impulse[0], impulse[1], impulse[2], 0]) / self.mass

            self.translation(self.velocity[0], self.velocity[1]+0.3, self.velocity[2])
            
            self.a_velocity -= np.cross(r_1, impulse) / 5

    def rotate_local_coordinate(self):
        for i in range(len(self.vertices)):
            r = self.vertices[i][:3] - self.center[:3]
            rotation_change = np.cross(self.a_velocity, r)
            self.vertices[i][:3] += rotation_change


class Plane(Shape):
    def __init__(self):
        self.mass = 10
        self.vertices = np.array([
            [-1., -1., 0, 1.],
            [1., -1., 0, 1.],
            [1., 1., 0, 1.],
            [-1., 1., 0, 1.]
        ])

        self.center = np.array([0., 0., 0., 1.])

        self.triangle = [
            [0, 3, 2],
            [0, 2, 1]
        ]
            

def project(vertex, camera_distance):
    x, y, z, w = vertex
    factor = FOV / (z + camera_distance) if z + camera_distance > 0 else 0
    x_2d = x * factor + WIDTH / 2
    y_2d = -y * factor + HEIGHT / 2
    return (x_2d, y_2d)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("3D Engine")
    clock = pygame.time.Clock()
    
    camera_distance = 5
    
    plane = Plane()

    plane.scale(3, 10, 3)
    plane.rotate(60, 60, 0)
    plane.translation(0, -5, 4)

    cube = Cube()
    cube.rotate(0, 0, 0)
    cube.translation(0, 20, 4)
    

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        cube.gravity()
        cube.handle_collision(plane)
        cube.rotate_local_coordinate()
        screen.fill((0, 0, 0))

        projected_plane = []

        for vertex in plane.vertices:
            projected_plane.append(project(np.array(vertex), camera_distance))

        for vertex in plane.mesh_represent():
            projected_plane.append(project(np.array(vertex), camera_distance))

        for vertex in projected_plane:
            if not (vertex[0] == WIDTH / 2 and vertex[1] == HEIGHT / 2):
                pygame.draw.circle(screen, (255, 0, 0), vertex, 3)
            
        projected_cube = []
        for vertex in cube.vertices:
            projected_cube.append(project(np.array(vertex), camera_distance))

        for vertex in projected_cube:
            if not (vertex[0] == WIDTH / 2 and vertex[1] == HEIGHT / 2):
                pygame.draw.circle(screen, (255,255,255), vertex, 3)
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
