"""Tiny feedforward neural network — numpy only, no ML library.

Architecture (sizes come from config):

    INPUT (5)  →  HIDDEN (8, tanh)  →  OUTPUT (2, tanh)

Weights are stored internally as plain numpy matrices for fast forward passes
but exposed as a single flat 1D numpy array via get_weights / set_weights.
A flat representation makes the genetic operations trivial: uniform crossover
is "pick element-wise from two parents", and per-weight mutation is "add
Gaussian noise at random indices".
"""

import numpy as np
import pygame

import config as C


class NeuralNetwork:
    """Two-layer feedforward network using tanh activations.

    Attributes:
        W1, b1, W2, b2: parameter matrices/vectors.
        input_act, hidden_act, output_act: cached activations from the most
            recent forward() call — read by draw() so the visualiser can pulse
            in real time.
    """

    # Order matches the SENSOR_ANGLES list in config (-90°, -45°, 0°, +45°, +90°).
    INPUT_LABELS = ["L90", "L45", "FWD", "R45", "R90"]
    OUTPUT_LABELS = ["STEER", "GAS"]

    def __init__(self, weights=None):
        """Build a network. If `weights` is None, randomise; otherwise load them."""
        self.n_in = C.NN_INPUT_SIZE
        self.n_hidden = C.NN_HIDDEN_SIZE
        self.n_out = C.NN_OUTPUT_SIZE

        # Parameter shapes. Stored in (input_dim, output_dim) layout so
        # forward pass is a clean matrix multiply: y = x @ W + b.
        self.W1 = np.zeros((self.n_in, self.n_hidden))
        self.b1 = np.zeros(self.n_hidden)
        self.W2 = np.zeros((self.n_hidden, self.n_out))
        self.b2 = np.zeros(self.n_out)

        # Most recent activations (for live visualisation).
        self.input_act = np.zeros(self.n_in)
        self.hidden_act = np.zeros(self.n_hidden)
        self.output_act = np.zeros(self.n_out)

        if weights is None:
            self.randomise()
        else:
            self.set_weights(weights)

    # ------------------------------------------------------------------
    # Weight management — the GA only ever touches these methods.
    # ------------------------------------------------------------------

    @property
    def weight_count(self):
        """Total number of trainable parameters (weights + biases)."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def randomise(self):
        """Re-initialise every weight to a small Gaussian sample."""
        sigma = C.NN_INIT_SIGMA
        self.W1 = np.random.randn(self.n_in, self.n_hidden) * sigma
        self.b1 = np.random.randn(self.n_hidden) * sigma
        self.W2 = np.random.randn(self.n_hidden, self.n_out) * sigma
        self.b2 = np.random.randn(self.n_out) * sigma

    def get_weights(self):
        """Return every parameter concatenated into one flat 1D array."""
        # ravel() returns flat views (cheap); concatenate copies them once.
        return np.concatenate([
            self.W1.ravel(),
            self.b1.ravel(),
            self.W2.ravel(),
            self.b2.ravel(),
        ])

    def set_weights(self, flat):
        """Load parameters from a flat 1D array, in the same order get_weights uses."""
        flat = np.asarray(flat, dtype=float)
        if flat.size != self.weight_count:
            raise ValueError(
                f"Expected {self.weight_count} weights, got {flat.size}"
            )
        idx = 0
        # Slice the flat vector chunk-by-chunk and reshape into each matrix.
        size = self.W1.size
        self.W1 = flat[idx:idx + size].reshape(self.n_in, self.n_hidden).copy()
        idx += size
        size = self.b1.size
        self.b1 = flat[idx:idx + size].copy()
        idx += size
        size = self.W2.size
        self.W2 = flat[idx:idx + size].reshape(self.n_hidden, self.n_out).copy()
        idx += size
        size = self.b2.size
        self.b2 = flat[idx:idx + size].copy()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, inputs):
        """Run one forward pass.

        Args:
            inputs: 1D iterable of length n_in (sensor readings in [0, 1]).
        Returns:
            1D numpy array of length n_out, each value in [-1, 1].
        """
        x = np.asarray(inputs, dtype=float)
        self.input_act = x
        # Hidden layer. tanh keeps values bounded so signals can't explode.
        self.hidden_act = np.tanh(x @ self.W1 + self.b1)
        # Output layer. tanh gives [-1, 1] which maps directly to (steer, throttle).
        self.output_act = np.tanh(self.hidden_act @ self.W2 + self.b2)
        return self.output_act

    # ------------------------------------------------------------------
    # Live visualisation
    # ------------------------------------------------------------------

    def draw(self, surface, rect, font=None):
        """Render a live diagram of the network into the given rect.

        Neurons are circles whose radius scales with |activation|. Connection
        lines are coloured green for positive weights / red for negative, with
        alpha proportional to |weight| relative to the largest weight in that
        layer.
        """
        x, y, w, h = rect
        # Draw onto an alpha surface so connection lines can be translucent
        # and the whole panel can have a semi-transparent dark background.
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        panel.fill((20, 20, 24, 220))

        # X-positions of the three neuron columns within the panel.
        col_x = [int(w * 0.20), int(w * 0.50), int(w * 0.78)]

        # Vertical positions, evenly spaced with a margin top/bottom.
        def positions(n):
            margin = int(h * 0.15)
            usable = h - 2 * margin
            if n == 1:
                return [margin + usable // 2]
            return [margin + int(i * usable / (n - 1)) for i in range(n)]

        in_y = positions(self.n_in)
        hid_y = positions(self.n_hidden)
        out_y = positions(self.n_out)

        # Normalise line opacity by the largest weight in each layer.
        max_w1 = max(float(np.abs(self.W1).max()), 1e-6)
        max_w2 = max(float(np.abs(self.W2).max()), 1e-6)

        # Connection lines first (so neurons render on top).
        for i in range(self.n_in):
            for j in range(self.n_hidden):
                self._draw_connection(
                    panel,
                    (col_x[0], in_y[i]),
                    (col_x[1], hid_y[j]),
                    self.W1[i, j], max_w1,
                )
        for j in range(self.n_hidden):
            for k in range(self.n_out):
                self._draw_connection(
                    panel,
                    (col_x[1], hid_y[j]),
                    (col_x[2], out_y[k]),
                    self.W2[j, k], max_w2,
                )

        # Neurons (circles).
        for i, py in enumerate(in_y):
            self._draw_neuron(panel, (col_x[0], py), self.input_act[i])
        for j, py in enumerate(hid_y):
            self._draw_neuron(panel, (col_x[1], py), self.hidden_act[j])
        for k, py in enumerate(out_y):
            self._draw_neuron(panel, (col_x[2], py), self.output_act[k])

        # Labels on the outside of the input / output columns.
        if font is None:
            font = pygame.font.SysFont("consolas", C.FONT_SIZE_LABEL)
        for i, py in enumerate(in_y):
            label = font.render(self.INPUT_LABELS[i], True, (220, 220, 220))
            panel.blit(label, (col_x[0] - 48, py - label.get_height() // 2))
        for k, py in enumerate(out_y):
            label = font.render(self.OUTPUT_LABELS[k], True, (220, 220, 220))
            panel.blit(label, (col_x[2] + 18, py - label.get_height() // 2))
            # Numeric output value so you can read what the car is doing.
            val = font.render(f"{self.output_act[k]:+.2f}", True, (255, 220, 120))
            panel.blit(val, (col_x[2] + 18, py + 4))

        surface.blit(panel, (x, y))

    def _draw_connection(self, surface, p1, p2, weight, max_weight):
        """Stroke one connection line, coloured by sign and opacity by magnitude."""
        # Weak weights still get a faint baseline alpha so the topology is visible.
        intensity = abs(weight) / max_weight
        alpha = int(30 + 200 * intensity)
        color = C.NN_POS_COLOR if weight >= 0 else C.NN_NEG_COLOR
        pygame.draw.line(surface, (*color, alpha), p1, p2, 1)

    def _draw_neuron(self, surface, pos, activation):
        """Draw one neuron as a filled circle sized by activation magnitude."""
        base_r = 6
        r = int(base_r + abs(float(activation)) * 8)
        # Green = positive activation, red = negative. Sensor inputs are in
        # [0, 1] so they are always green; hidden/output may flip.
        color = (90, 220, 110) if activation >= 0 else (230, 90, 90)
        pygame.draw.circle(surface, color, pos, r)
        # Thin outline so circles remain visible against bright backgrounds.
        pygame.draw.circle(surface, (240, 240, 240), pos, r, 1)
