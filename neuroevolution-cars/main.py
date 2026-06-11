"""Entry point: pygame game loop driving the neuroevolution simulation.

Run with:
    python main.py

See README.md for keyboard controls and a tour of how the project works.
"""

import sys
import pygame

import config as C
import ui
from track import Track
from evolution import Evolution


def run():
    """Launch the simulation and run until the user closes the window."""
    pygame.init()
    pygame.display.set_caption("Neuroevolution Cars")
    screen = pygame.display.set_mode((C.WINDOW_WIDTH, C.WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    hud_font = pygame.font.SysFont("consolas", C.FONT_SIZE_HUD)
    label_font = pygame.font.SysFont("consolas", C.FONT_SIZE_LABEL)

    # ---- World ----
    track = Track()
    evo = Evolution()
    cars = evo.spawn_initial_population(track)

    # ---- Per-run state ----
    frame_in_gen = 0
    sim_speed_index = 0   # index into C.SIM_SPEEDS
    paused = False
    show_nn = True

    # Fixed rectangles for the side panels.
    history_rect = (C.WINDOW_WIDTH - 250, 10, 240, 130)
    nn_rect = (C.WINDOW_WIDTH - 430, C.WINDOW_HEIGHT - 250, 420, 220)

    running = True
    while running:
        # ------------------------------------------------------------------
        # Events / input
        # ------------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT:
                    # Force the next generation immediately.
                    cars = evo.new_generation(cars, track)
                    frame_in_gen = 0
                elif event.key == pygame.K_n:
                    show_nn = not show_nn
                elif event.key == pygame.K_r:
                    # Full reset: gen 1, fresh random weights.
                    evo.reset()
                    cars = evo.spawn_initial_population(track)
                    frame_in_gen = 0
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    idx = event.key - pygame.K_1
                    if idx < len(C.SIM_SPEEDS):
                        sim_speed_index = idx

        sim_speed = C.SIM_SPEEDS[sim_speed_index]

        # ------------------------------------------------------------------
        # Physics: run `sim_speed` ticks per render frame so visible speed
        # can scale without changing the actual frame-rate.
        # ------------------------------------------------------------------
        if not paused:
            for _ in range(sim_speed):
                any_alive = False
                for car in cars:
                    if car.alive:
                        car.update(track)
                        if car.alive:
                            any_alive = True
                frame_in_gen += 1
                # End the generation if everyone has crashed or we've hit
                # the per-generation time cap.
                if not any_alive or frame_in_gen >= C.MAX_GEN_FRAMES:
                    cars = evo.new_generation(cars, track)
                    frame_in_gen = 0
                    # Stop additional inner ticks this frame so the render
                    # below clearly shows the new generation starting fresh.
                    break

        # ------------------------------------------------------------------
        # Pick the best live car (for sensor display + NN panel)
        # ------------------------------------------------------------------
        alive_cars = [c for c in cars if c.alive]
        if alive_cars:
            best_alive = max(alive_cars, key=lambda c: c.fitness)
        else:
            # Everyone is dead — fall back to overall best so the NN panel
            # still has something meaningful to display.
            best_alive = max(cars, key=lambda c: c.fitness)

        # Map every live car to its current fitness rank (0 = best).
        ranked_alive = sorted(alive_cars, key=lambda c: c.fitness, reverse=True)
        rank_lookup = {id(c): i for i, c in enumerate(ranked_alive)}

        # ------------------------------------------------------------------
        # Render
        # ------------------------------------------------------------------
        track.draw(screen)
        # Dead cars first (dim), so live cars are clearly on top.
        for car in cars:
            if not car.alive:
                car.draw(screen, ui.color_for_car(0, alive=False))
        # Live cars from worst to best so the leader ends up on top.
        for car in reversed(ranked_alive):
            rank = rank_lookup[id(car)]
            color = ui.color_for_car(rank, alive=True)
            car.draw(screen, color, show_sensors=(car is best_alive))

        # HUD overlays.
        ui.draw_hud(
            screen, evo, len(alive_cars), len(cars),
            sim_speed, paused, show_nn, hud_font,
        )
        ui.draw_history_graph(screen, evo.history, history_rect, label_font)
        if show_nn:
            ui.draw_nn_panel(screen, best_alive.nn, nn_rect, label_font)
        ui.draw_controls_hint(screen, label_font)

        pygame.display.flip()
        clock.tick(C.FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run()
