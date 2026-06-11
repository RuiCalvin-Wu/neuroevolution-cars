"""Stateless UI helpers: HUD, history graph, neural-network panel, car colours.

All functions take a `surface` and draw directly onto it. No module-level
state is kept — main.py owns the simulation, this module just renders it.
"""

import pygame

import config as C


# ---------------------------------------------------------------------------
# Colour selection
# ---------------------------------------------------------------------------

def color_for_car(rank, alive):
    """Pick a render colour for a car based on its current rank.

    Args:
        rank: 0 for the leading (highest-fitness) live car, 1..N for the rest.
        alive: False if the car has crashed (drawn dim).
    Returns:
        (R, G, B) tuple.
    """
    if not alive:
        return C.COLOR_DEAD
    if rank == 0:
        return C.COLOR_BEST
    if rank < C.TOP_HIGHLIGHT_COUNT:
        return C.COLOR_TOP
    return C.COLOR_NORMAL


# ---------------------------------------------------------------------------
# HUD panel (top-left)
# ---------------------------------------------------------------------------

def draw_hud(surface, evo, alive_count, total_count, sim_speed, paused, show_nn, font):
    """Render the top-left stats panel."""
    lines = [
        f"Gen          {evo.generation}",
        f"Alive        {alive_count} / {total_count}",
        f"Best (gen)   {evo.best_fitness_this_gen:.3f}",
        f"Best (ever)  {evo.best_fitness_ever:.3f}",
        f"Avg (last)   {evo.avg_fitness:.3f}",
        f"Sim speed    {sim_speed}x",
    ]
    if paused:
        lines.append("** PAUSED **")
    if not show_nn:
        lines.append("(N to show NN)")

    # Translucent dark panel sized to fit the text.
    padding = 10
    line_h = font.get_linesize()
    panel_w = 220
    panel_h = padding * 2 + line_h * len(lines)
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((15, 15, 20, 200))
    for i, line in enumerate(lines):
        # Highlight the "** PAUSED **" line in yellow so it stands out.
        color = (255, 220, 80) if "PAUSED" in line else (230, 230, 240)
        text = font.render(line, True, color)
        panel.blit(text, (padding, padding + i * line_h))
    surface.blit(panel, (10, 10))


# ---------------------------------------------------------------------------
# Fitness history graph (top-right)
# ---------------------------------------------------------------------------

def draw_history_graph(surface, history, rect, font):
    """Plot the best-fitness-per-generation history as a small line chart."""
    x, y, w, h = rect
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((15, 15, 20, 200))

    # Title.
    title = font.render("Best fitness / gen", True, (230, 230, 240))
    panel.blit(title, (8, 4))

    # Inner plotting area.
    pad_l, pad_r, pad_t, pad_b = 30, 8, 24, 18
    plot_x, plot_y = pad_l, pad_t
    plot_w, plot_h = w - pad_l - pad_r, h - pad_t - pad_b

    # Frame around the plot.
    pygame.draw.rect(panel, (60, 60, 70), (plot_x, plot_y, plot_w, plot_h), 1)

    if len(history) >= 1:
        max_val = max(max(history), 1.0)
        # Y-axis label (max value).
        y_label = font.render(f"{max_val:.2f}", True, (180, 180, 190))
        panel.blit(y_label, (4, plot_y - 4))
        # X-axis label (gen count).
        x_label = font.render(f"last {len(history)} gen", True, (180, 180, 190))
        panel.blit(x_label, (plot_x, plot_y + plot_h + 2))

        # Convert history values to screen coordinates inside the plot area.
        if len(history) >= 2:
            points = []
            for i, v in enumerate(history):
                px = plot_x + int(i * plot_w / max(1, len(history) - 1))
                py = plot_y + plot_h - int(v / max_val * plot_h)
                points.append((px, py))
            pygame.draw.lines(panel, (120, 220, 140), False, points, 2)
            # Mark the most recent point with a dot.
            pygame.draw.circle(panel, (255, 220, 80), points[-1], 3)
        else:
            # Just one data point so far — render it as a single dot.
            v = history[0]
            px = plot_x + plot_w // 2
            py = plot_y + plot_h - int(v / max_val * plot_h)
            pygame.draw.circle(panel, (255, 220, 80), (px, py), 3)

    surface.blit(panel, (x, y))


# ---------------------------------------------------------------------------
# Neural-network panel (bottom)
# ---------------------------------------------------------------------------

def draw_nn_panel(surface, nn, rect, font):
    """Delegate to the network's own draw method."""
    if nn is None:
        return
    nn.draw(surface, rect, font=font)


# ---------------------------------------------------------------------------
# Controls hint (bottom-left, single line)
# ---------------------------------------------------------------------------

def draw_controls_hint(surface, font):
    """Print a tiny one-line reminder of the keyboard controls."""
    text = "SPACE pause   → next gen   1-5 sim speed   N toggle NN   R reset"
    rendered = font.render(text, True, (200, 200, 210))
    surface.blit(rendered, (10, C.WINDOW_HEIGHT - rendered.get_height() - 6))
