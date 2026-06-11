"""Central configuration for the neuroevolution-cars project.

Every tunable value lives here so you can experiment without hunting through
the rest of the codebase. Comments above each constant explain what it does
and the realistic range to play with.
"""

import math

# ---------------------------------------------------------------------------
# Window / rendering
# ---------------------------------------------------------------------------

# Window dimensions in pixels.
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

# Target render frame-rate. Physics may tick multiple times per frame depending
# on SIM_SPEED, but the screen is always redrawn at this rate.
FPS = 60

# Available simulation speed multipliers. Each multiplier means "run N physics
# ticks per render frame". Press number keys 1..5 to switch.
SIM_SPEEDS = [1, 2, 5, 10, 20]

# Background colour of the window (R, G, B).
BG_COLOR = (24, 24, 28)

# ---------------------------------------------------------------------------
# Track
# ---------------------------------------------------------------------------

# Half-width of the drivable road in pixels. The track is built by offsetting
# the centerline by ±TRACK_HALF_WIDTH on each side.
TRACK_HALF_WIDTH = 40

# Road fill colour and centerline / boundary colours.
TRACK_ROAD_COLOR = (60, 60, 65)
TRACK_EDGE_COLOR = (200, 200, 210)
TRACK_CENTERLINE_COLOR = (220, 220, 90)
TRACK_START_LINE_COLOR = (220, 60, 60)

# Number of pixels in one dash (and one gap) of the dashed centerline.
TRACK_DASH_LENGTH = 12

# ---------------------------------------------------------------------------
# Car physics
# ---------------------------------------------------------------------------

# Top forward speed in pixels per physics tick.
CAR_MAX_SPEED = 6.0

# How much speed is gained per tick when throttle = +1.0.
CAR_ACCELERATION = 0.18

# Multiplicative drag applied every tick (0.95 = lose 5% of speed per tick).
CAR_FRICTION = 0.96

# Max steering change in radians per tick at zero speed. Scales down with
# speed so a fast car cannot pivot in place.
CAR_TURN_SPEED = 0.08

# Dimensions of the rendered car rectangle (length × width, in pixels).
CAR_LENGTH = 18
CAR_WIDTH = 10

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------

# Number of distance sensors fanned out around the car heading.
SENSOR_COUNT = 5

# Angles (radians) of each sensor relative to the car's forward direction.
# -90°, -45°, 0°, +45°, +90° gives a 180° forward fan.
SENSOR_ANGLES = [
    math.radians(-90),
    math.radians(-45),
    math.radians(0),
    math.radians(45),
    math.radians(90),
]

# Maximum distance a sensor ray will travel before giving up (pixels).
SENSOR_MAX_DIST = 200

# Step size in pixels along the ray when raycasting. Smaller = more accurate
# but slower. 4 is a good balance.
SENSOR_STEP = 4

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------

# Input size = SENSOR_COUNT. Output size = 2 (steer, throttle). Both fixed.
NN_INPUT_SIZE = SENSOR_COUNT
NN_HIDDEN_SIZE = 8
NN_OUTPUT_SIZE = 2

# Standard deviation used when generating fresh random weights.
NN_INIT_SIGMA = 1.0

# ---------------------------------------------------------------------------
# Genetic algorithm
# ---------------------------------------------------------------------------

# Number of cars per generation.
POPULATION_SIZE = 30

# Fraction of the population (sorted by fitness) treated as "elite" — these
# are the parents used to seed the next generation. 0.2 = top 20%.
ELITE_FRACTION = 0.2

# Number of best-of-generation cars copied unchanged into the next generation
# (true elitism — guarantees performance never regresses).
ELITISM_COUNT = 2

# Probability that an individual weight is mutated.
MUTATION_RATE = 0.10

# Standard deviation of the Gaussian noise added to a mutated weight.
MUTATION_SIGMA = 0.30

# Hard cap on how many physics ticks a single generation may run before being
# forced to end. Without this, one survivor can stall progress indefinitely.
MAX_GEN_FRAMES = 60 * 30  # ~30 seconds at 1× speed

# ---------------------------------------------------------------------------
# UI / HUD
# ---------------------------------------------------------------------------

# Colours for the rank-coloured cars.
COLOR_BEST = (255, 230, 60)          # bright yellow — single best car
COLOR_TOP = (255, 150, 50)           # orange — top 5
COLOR_NORMAL = (180, 180, 190)       # dim white — everyone else alive
COLOR_DEAD = (70, 70, 80)            # very dim — crashed cars

# How many of the top-ranked cars get the "orange" highlight (excluding #1).
TOP_HIGHLIGHT_COUNT = 5

# How many generations of history to remember for the fitness graph.
HISTORY_LENGTH = 20

# Font sizes for HUD text.
FONT_SIZE_HUD = 18
FONT_SIZE_LABEL = 12

# Colour of the connection lines in the NN visualizer (positive / negative).
NN_POS_COLOR = (90, 220, 110)
NN_NEG_COLOR = (230, 90, 90)
