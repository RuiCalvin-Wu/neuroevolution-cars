"""Simulated car: pose, physics, sensors, fitness, and rendering.

Each car owns one NeuralNetwork that maps 5 sensor distances to (steer,
throttle) commands. Physics is integrated once per `update()` call. Cars
die on wall collision and their fitness is computed from arc-length
progress along the track centerline (lap-aware).
"""

import math
import pygame

import config as C
from neural_network import NeuralNetwork


class Car:
    """A self-driving simulated car.

    Attributes:
        x, y: position in pixels.
        angle: heading in radians (0 = +x).
        speed: scalar speed along the heading (pixels per tick, may be negative).
        alive: False once the car has crashed.
        fitness: lap-aware progress; monotonically non-decreasing.
        nn: NeuralNetwork that produces (steer, throttle) each tick.
        last_sensors: most recent normalised sensor readings, for drawing.
    """

    def __init__(self, x, y, angle, nn=None):
        """Spawn a car. If `nn` is None, the car gets a fresh random network."""
        self.x = float(x)
        self.y = float(y)
        self.angle = float(angle)
        self.speed = 0.0
        self.alive = True
        # Fitness state. We track lap count separately from arc progress so
        # that crossing the start/finish line correctly counts as +1 lap.
        self.fitness = 0.0
        self.laps = 0
        self.last_progress = 0.0
        # The network whose weights are evolved by the GA.
        self.nn = nn if nn is not None else NeuralNetwork()
        # Cache of the most recent sensor readings so draw() can show them
        # without having to raycast again.
        self.last_sensors = [1.0] * C.SENSOR_COUNT

    # ------------------------------------------------------------------
    # Physics tick
    # ------------------------------------------------------------------

    def update(self, track):
        """Run one physics step. Does nothing if the car has already crashed."""
        if not self.alive:
            return

        # 1. Sense: cast 5 rays and read normalised distances.
        self.last_sensors = self._read_sensors(track)

        # 2. Think: forward pass through the network.
        steer, throttle = self.nn.forward(self.last_sensors)

        # 3. Act: apply throttle, friction, clamp to top speed.
        self.speed += float(throttle) * C.CAR_ACCELERATION
        self.speed *= C.CAR_FRICTION
        if self.speed > C.CAR_MAX_SPEED:
            self.speed = C.CAR_MAX_SPEED
        elif self.speed < -C.CAR_MAX_SPEED:
            self.speed = -C.CAR_MAX_SPEED

        # 4. Steer. Scale turn rate with current speed so a stationary car
        #    cannot pivot in place (also rules out the trivial "spin forever"
        #    strategy that some random networks would otherwise discover).
        speed_factor = min(1.0, abs(self.speed) / (C.CAR_MAX_SPEED * 0.3))
        self.angle += float(steer) * C.CAR_TURN_SPEED * speed_factor

        # 5. Move along the new heading.
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # 6. Collide / score.
        if track.check_collision(self.x, self.y):
            self.alive = False
            return
        self._update_fitness(track)

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def _read_sensors(self, track):
        """Cast all SENSOR_COUNT rays and return a list of normalised distances."""
        return [self._cast_ray(track, off) for off in C.SENSOR_ANGLES]

    def _cast_ray(self, track, angle_offset):
        """Cast a single ray from the car's centre, return distance / max_dist.

        Marches outward in SENSOR_STEP-pixel increments until the track mask
        reports off-road or SENSOR_MAX_DIST is reached. Stepping is a simple
        and adequate substitute for analytic ray-vs-polygon intersection on a
        per-pixel-mask track.
        """
        ray_angle = self.angle + angle_offset
        dx = math.cos(ray_angle) * C.SENSOR_STEP
        dy = math.sin(ray_angle) * C.SENSOR_STEP
        px, py = self.x, self.y
        distance = 0.0
        while distance < C.SENSOR_MAX_DIST:
            px += dx
            py += dy
            distance += C.SENSOR_STEP
            if track.check_collision(px, py):
                break
        # 0.0 = wall right at the car, 1.0 = nothing within sensor range.
        return min(distance / C.SENSOR_MAX_DIST, 1.0)

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _update_fitness(self, track):
        """Update lap-aware fitness from the track's arc-length progress.

        `track.get_progress` returns a value in [0, 1] that wraps around at
        the start/finish line. We watch for big jumps in that value to detect
        when the car has crossed the line in either direction, and bump or
        decrement a lap counter accordingly. Fitness uses max() so any
        backwards motion can never reduce a car's score.
        """
        new_progress = track.get_progress(self.x, self.y)
        delta = new_progress - self.last_progress
        if delta < -0.5:
            # Progress jumped from ~1.0 down to ~0.0 — a forward lap.
            self.laps += 1
        elif delta > 0.5:
            # Progress jumped from ~0.0 up to ~1.0 — backward lap (rare).
            self.laps -= 1
        self.last_progress = new_progress
        candidate = self.laps + new_progress
        if candidate > self.fitness:
            self.fitness = candidate

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, surface, color, show_sensors=False):
        """Render the car. If `show_sensors`, also draw its 5 ray casts."""
        # Rotate the four local-frame corners into world coordinates.
        cx, cy = self.x, self.y
        half_l = C.CAR_LENGTH / 2.0
        half_w = C.CAR_WIDTH / 2.0
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        corners_local = [
            ( half_l, -half_w),
            ( half_l,  half_w),
            (-half_l,  half_w),
            (-half_l, -half_w),
        ]
        corners = [
            (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)
            for lx, ly in corners_local
        ]
        pygame.draw.polygon(surface, color, corners)
        # A small dot poking out the front so heading is visually obvious.
        nose_x = cx + cos_a * (half_l + 3)
        nose_y = cy + sin_a * (half_l + 3)
        pygame.draw.circle(surface, color, (int(nose_x), int(nose_y)), 2)

        if show_sensors:
            self._draw_sensors(surface)

    def _draw_sensors(self, surface):
        """Draw the most recently-cast sensor rays as faint yellow lines."""
        for angle_offset, normalised_dist in zip(C.SENSOR_ANGLES, self.last_sensors):
            ray_angle = self.angle + angle_offset
            length = normalised_dist * C.SENSOR_MAX_DIST
            ex = self.x + math.cos(ray_angle) * length
            ey = self.y + math.sin(ray_angle) * length
            pygame.draw.line(
                surface, (240, 240, 100), (self.x, self.y), (ex, ey), 1
            )
            # Small dot at the ray endpoint (either a wall hit or max range).
            pygame.draw.circle(surface, (255, 100, 100), (int(ex), int(ey)), 2)
