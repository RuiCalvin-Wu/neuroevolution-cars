"""Track definition, rendering, collision detection, and progress measurement.

The track is a closed loop defined by hand-authored centerline waypoints. The
drivable road is built by drawing thick line segments between consecutive
waypoints; a `pygame.mask.Mask` derived from that pre-rendered surface is
used for fast per-pixel collision queries.
"""

import math
import pygame

import config as C


class Track:
    """A looping race track with sensor-friendly per-pixel collision.

    Attributes:
        waypoints: List of (x, y) centerline points forming a closed loop
            (the last point implicitly connects back to the first).
        total_length: Sum of all segment lengths — used to normalise progress.
        start_x, start_y, start_angle: Pose at which all cars spawn.
        road_surface: Pre-rendered opaque surface, blitted each frame.
        collision_mask: Bitmask where 1 = on-road, 0 = off-road.
    """

    def __init__(self):
        # Centerline waypoints (clockwise loop). Includes:
        #   - a long bottom straight
        #   - a sharp 90°-ish corner at bottom-right
        #   - a long sweep up the right side
        #   - a chicane (S-curves) along the top
        #   - a smooth top-left corner and left straight back to start
        self.waypoints = [
            (200, 680),   # start (facing right along +x)
            (450, 680),
            (750, 680),
            (950, 680),
            (1100, 600),  # sharp corner mid
            (1160, 460),
            (1160, 300),  # right-side straight
            (1100, 200),  # top-right corner
            (960, 240),   # chicane: down
            (820, 180),   # chicane: up
            (680, 240),   # chicane: down
            (540, 180),   # chicane: up
            (400, 240),   # chicane exit
            (220, 220),   # top-left corner
            (140, 350),
            (140, 550),
            (180, 660),
        ]

        # Pre-compute the length of each segment and the cumulative arc length
        # from the start. cumulative_lengths[i] = arc length at waypoint i.
        n = len(self.waypoints)
        self.segment_lengths = []
        self.cumulative_lengths = [0.0]
        for i in range(n):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[(i + 1) % n]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            self.segment_lengths.append(seg_len)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + seg_len)
        self.total_length = self.cumulative_lengths[-1]

        # Spawn pose: at first waypoint, heading toward the second.
        sx, sy = self.waypoints[0]
        nx, ny = self.waypoints[1]
        self.start_x = float(sx)
        self.start_y = float(sy)
        self.start_angle = math.atan2(ny - sy, nx - sx)

        # Pre-render the road and build the collision mask from it.
        self.road_surface = self._build_road_surface()
        mask_source = self.road_surface.copy()
        # Treat background pixels as "transparent" so the mask only contains
        # the road. mask.get_at returns 1 where the road was drawn.
        mask_source.set_colorkey(C.BG_COLOR)
        self.collision_mask = pygame.mask.from_surface(mask_source)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def get_start_pose(self):
        """Return (x, y, angle) at which a new car should spawn."""
        return self.start_x, self.start_y, self.start_angle

    def check_collision(self, x, y):
        """Return True if (x, y) is OFF the road (or out of bounds)."""
        ix, iy = int(x), int(y)
        if ix < 0 or ix >= C.WINDOW_WIDTH or iy < 0 or iy >= C.WINDOW_HEIGHT:
            return True
        # mask.get_at returns 1 if the pixel is "set" (= road). Off road → 0.
        return self.collision_mask.get_at((ix, iy)) == 0

    def get_progress(self, x, y):
        """Return arc-length progress in [0, 1] along the centerline.

        Works by projecting (x, y) onto every segment, picking the segment
        whose projection is closest to the point, and converting the
        projection's position on that segment to a fraction of the total
        lap length.
        """
        n = len(self.waypoints)
        best_dist_sq = float("inf")
        best_arc = 0.0
        for i in range(n):
            x1, y1 = self.waypoints[i]
            x2, y2 = self.waypoints[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq == 0:
                continue
            # Project the point onto the (infinite) line through the segment,
            # then clamp t to [0, 1] to stay on the segment itself.
            t = ((x - x1) * dx + (y - y1) * dy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            px, py = x1 + t * dx, y1 + t * dy
            d_sq = (x - px) ** 2 + (y - py) ** 2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_arc = self.cumulative_lengths[i] + t * self.segment_lengths[i]
        return best_arc / self.total_length

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _build_road_surface(self):
        """Render the road once into an opaque surface used for collisions and display."""
        surf = pygame.Surface((C.WINDOW_WIDTH, C.WINDOW_HEIGHT))
        surf.fill(C.BG_COLOR)
        thickness = C.TRACK_HALF_WIDTH * 2
        n = len(self.waypoints)
        # Thick lines between consecutive waypoints form the road. Circles at
        # each joint fill the wedge gaps that lines leave between segments.
        for i in range(n):
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % n]
            pygame.draw.line(surf, C.TRACK_ROAD_COLOR, p1, p2, thickness)
        for p in self.waypoints:
            pygame.draw.circle(surf, C.TRACK_ROAD_COLOR, p, C.TRACK_HALF_WIDTH)
        return surf

    def draw(self, surface):
        """Blit the cached road, then overlay dashed centerline and start line."""
        surface.blit(self.road_surface, (0, 0))
        n = len(self.waypoints)
        for i in range(n):
            self._draw_dashed_segment(
                surface, self.waypoints[i], self.waypoints[(i + 1) % n]
            )
        self._draw_start_line(surface)

    def _draw_dashed_segment(self, surface, p1, p2):
        """Stroke a dashed line from p1 to p2 along the centerline."""
        x1, y1 = p1
        x2, y2 = p2
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        dash = C.TRACK_DASH_LENGTH
        # Unit vector along the segment.
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        # Step in chunks of "dash on + dash off" along the segment.
        n_dashes = int(length // (dash * 2))
        for i in range(n_dashes + 1):
            sx = x1 + ux * dash * 2 * i
            sy = y1 + uy * dash * 2 * i
            ex = sx + ux * dash
            ey = sy + uy * dash
            pygame.draw.line(
                surface, C.TRACK_CENTERLINE_COLOR, (sx, sy), (ex, ey), 2
            )

    def _draw_start_line(self, surface):
        """Draw a red bar perpendicular to the first segment at the start point."""
        x1, y1 = self.waypoints[0]
        x2, y2 = self.waypoints[1]
        dx, dy = x2 - x1, y2 - y1
        seg_len = math.hypot(dx, dy)
        if seg_len == 0:
            return
        # Perpendicular = 90° rotation of the segment direction.
        nx, ny = -dy / seg_len, dx / seg_len
        half = C.TRACK_HALF_WIDTH
        p_a = (x1 + nx * half, y1 + ny * half)
        p_b = (x1 - nx * half, y1 - ny * half)
        pygame.draw.line(surface, C.TRACK_START_LINE_COLOR, p_a, p_b, 4)
