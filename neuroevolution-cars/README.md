# Neuroevolution Cars

A small from-scratch Python project where 30 simulated cars learn to drive a
2D race track using tiny feedforward neural networks evolved by a genetic
algorithm — **no PyTorch, no TensorFlow, no labelled training data**. The
only dependencies are `pygame` for the window and `numpy` for matrix maths.

The first generation crashes within a second. By generation ~15, multiple
cars are running clean consistent laps. Watch the live neural-network panel
in the bottom-right to see which sensors are firing and what the brain is
telling the car to do at every moment.

---

## 1. Install and run

You need Python **3.10+**.

```bash
pip install pygame numpy
python main.py
```

A 1280×800 window opens, the track appears, and a generation of 30 cars
spawns at the start line.

---

## 2. Keyboard controls

| Key | What it does |
|-----|--------------|
| `SPACE` | Pause / unpause the simulation |
| `→` (Right arrow) | Skip to the next generation immediately |
| `1` `2` `3` `4` `5` | Simulation speed: 1×, 2×, 5×, 10×, 20× |
| `N` | Toggle the neural-network visualiser panel |
| `R` | Full reset — back to generation 1 with brand-new random weights |
| `ESC` | Quit |

Simulation speed multiplies how many physics ticks run per render frame —
rendering itself always stays at a smooth 60 FPS. So pressing `5` makes the
cars move 5× faster on screen while the window still feels responsive.

---

## 3. What is neuroevolution?

Most neural networks you read about are trained with **supervised learning**:
you give the network lots of labelled examples (an input + the right output)
and a *gradient-based optimiser* nudges the weights toward producing the
labelled outputs.

Neuroevolution is different. There are **no labels** and **no gradients**.
Instead:

1. Start with a *population* of networks, each with random weights.
2. Let every network try the task (here: drive the car).
3. Score each one with a **fitness function** (here: how far around the
   track did the car get?).
4. Take the best performers, mix and slightly perturb their weights to
   produce a new population, and repeat.

This works whenever you can *measure how well a network is doing* but
*can't write down the correct output for every input*. Driving is a perfect
example — there's no "correct steering angle" labelled per pixel, but it's
easy to score a car on how far it got before crashing.

Three ideas make this approach learn fast:

- **Elitism** — copy the top performers unchanged into the next generation,
  so the best behaviour never gets lost.
- **Crossover** — child networks combine weights from two strong parents,
  exploring new "hybrid brains".
- **Mutation** — small random changes to a few weights provide the
  innovation needed to discover behaviour that no parent had.

---

## 4. How the neural network works

Each car has its own little brain ([neural_network.py](neural_network.py))
with this shape:

```
INPUT (5)  →  HIDDEN (8, tanh)  →  OUTPUT (2, tanh)
```

**Inputs** (one per distance sensor):

| Index | Label | Sensor angle (relative to heading) |
|-------|-------|------------------------------------|
| 0 | `L90` | −90° (hard left) |
| 1 | `L45` | −45° (front-left) |
| 2 | `FWD` |   0° (straight ahead) |
| 3 | `R45` | +45° (front-right) |
| 4 | `R90` | +90° (hard right) |

Each input is a distance in `[0, 1]`: **0** means "wall right against the
car", **1** means "no wall within the 200-pixel sensor range".

**Outputs:**

| Index | Label | Meaning |
|-------|-------|---------|
| 0 | `STEER` | −1 = hard left, 0 = straight, +1 = hard right |
| 1 | `GAS` | −1 = reverse / brake, 0 = coast, +1 = full throttle |

**Activation function — `tanh`.**
`tanh(x)` squashes any real number into `(-1, 1)`. We use it on both layers
because:
- It naturally bounds the outputs to `[-1, 1]`, which maps directly to
  steering and throttle commands.
- It allows *negative* activations (unlike e.g. ReLU), which means a
  single neuron can both push and pull on a downstream weight, giving the
  network more expressive power per parameter.

**Weight storage.** Internally the network keeps weights as four numpy
arrays: `W1 (5×8)`, `b1 (8)`, `W2 (8×2)`, `b2 (2)` — **66 parameters**
total. But `get_weights()` flattens all of them into a single 1D array and
`set_weights(flat)` reverses that. This makes the genetic operators
gloriously simple: a flat array is easy to mix element-wise (crossover) or
perturb at random indices (mutation).

---

## 5. How the genetic algorithm works

Implemented in [evolution.py](evolution.py). Each generation:

1. **Score every car** by its lap-aware distance along the track
   centerline (see "Fitness" below).
2. **Rank** them best-to-worst.
3. **Elitism** — the top `ELITISM_COUNT` cars (default 2) get copied
   unchanged into the next generation.
4. **Selection** — the top `ELITE_FRACTION` (default 20%) become the
   "breeding pool".
5. **Fill the rest** by repeating, for each empty slot:
    - Pick **two random parents** from the breeding pool.
    - **Uniform crossover**: for every weight, randomly take it from
      either parent (50/50).
    - **Mutation**: with probability `MUTATION_RATE` per weight, add
      Gaussian noise of standard deviation `MUTATION_SIGMA`.
6. **Spawn** 30 fresh cars at the start line, each driven by one of the
   new weight vectors.

**Fitness = lap-aware arc-length progress.** A naive fitness like "how
far did the car move from the start point?" would reward cars for driving
in big circles near the start. Instead we measure the car's projection
onto the track centerline — so fitness only goes up when the car makes
progress *along the track*. Crossing the start/finish line forward bumps a
lap counter, and `fitness = laps + current_progress`. The car's fitness
also uses `max()` over time, so going backwards never *reduces* it (but
also doesn't reward it).

---

## 6. Tuning the simulation

All knobs live in [config.py](config.py) with comments. The ones worth
experimenting with:

| Constant | Default | What happens if you change it |
|----------|---------|-------------------------------|
| `POPULATION_SIZE` | 30 | Bigger = more diverse exploration, slower per gen. 60–100 finds clean drivers slightly faster on wall-clock time per gen but each gen costs more. |
| `ELITE_FRACTION` | 0.20 | Higher = stronger pressure (faster convergence but more risk of getting stuck in a local optimum). Try 0.1 or 0.4. |
| `ELITISM_COUNT` | 2 | Number of best individuals copied unchanged. 0 disables elitism — fitness can then drop between generations. |
| `MUTATION_RATE` | 0.10 | Fraction of weights mutated per child. Higher = more exploration, more noise. Try 0.05 or 0.20. |
| `MUTATION_SIGMA` | 0.30 | Size of each mutation. Lower = fine-tuning, slow improvement. Higher = bigger leaps, more crashes. |
| `NN_HIDDEN_SIZE` | 8 | Brain capacity. 4 is enough for simple tracks; 16 helps with complex chicanes but slows mutation convergence. |
| `SENSOR_COUNT` / `SENSOR_ANGLES` | 5 fan | Reducing to 3 sensors makes the problem much harder. Adding a back sensor hardly helps. |
| `SENSOR_MAX_DIST` | 200 | Longer = car sees obstacles earlier (easier learning), but at extremes inputs lose discrimination. |
| `CAR_MAX_SPEED` | 6.0 | Faster cars need to learn to brake before corners — much harder. |
| `MAX_GEN_FRAMES` | 1800 (~30s) | If you raise `CAR_MAX_SPEED` you may need to lower this so generations don't drag on. |

---

## 7. Project layout

```
neuroevolution-cars/
├── main.py              ← pygame loop, input, render order
├── car.py               ← car physics + sensor raycasting
├── neural_network.py    ← feedforward NN (numpy only) + visualiser
├── evolution.py         ← GA: selection, crossover, mutation, elitism
├── track.py             ← waypoints, road mask, arc-length progress
├── ui.py                ← HUD, fitness graph, car-colour helpers
├── config.py            ← every tunable constant
└── README.md            ← this file
```

The dependency arrow always points "down" through this list — `main.py`
imports almost everything, `config.py` imports nothing.

---

## 8. Ideas for extending

- **Save / load the best brain.** Pickle `best_car.nn.get_weights()` at the
  end of a run and load it back in `main.py` to skip the early generations.
- **Multiple tracks.** Make `Track` accept a waypoint list as an argument,
  define several, and rotate through them across generations so the
  evolved brains generalise.
- **Add a human-controlled car** that races against the leaders — wire the
  arrow keys directly to `Car.update()` bypassing the network.
- **Adaptive mutation.** Decay `MUTATION_SIGMA` as average fitness climbs
  — start at 0.5 for exploration, ease down toward 0.05 for fine-tuning.
- **NEAT** (NeuroEvolution of Augmenting Topologies). Instead of fixed
  5-8-2 architecture, let the GA evolve the network *structure* itself —
  adding new neurons and connections. Much more code but dramatically
  better at hard tasks.
- **Track editor.** Let the user place waypoints by clicking and rebuild
  the track mask on the fly.
- **Per-car colour by lineage** so you can see which parent's children
  dominate a generation.

---

## 9. What success looks like

Roughly:

| Gen | What you should see |
|-----|----------------------|
| 1   | All 30 cars crash almost immediately. |
| 3–5 | A few survivors emerge, hugging the inside of corners awkwardly. |
| 8–12 | First completed laps. |
| 15–20 | Multiple cars running clean, consistent laps. The fitness graph trends upward and the NN visualiser shows clearly differentiated activations on every corner. |

If it stalls (no improvement for many generations), try `R` to reset and
let it try again — random initialisation has a real effect on how fast a
good lineage gets going. Or tweak `MUTATION_SIGMA` upward.
