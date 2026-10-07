# Geometry Dash AI — a blind reinforcement learning agent

🇬🇧 English · 🇫🇷 [Français](README.fr.md)

An agent that learns to beat **official Geometry Dash levels in the real game**, by trial and error,
**without ever seeing the obstacles**. It only knows where it is in the level and how it is moving;
everything else, it learns by dying.

## Results

**16 of the 22 official levels beaten**, including the two demons tried so far, **Clubstep** and **Deadlocked**.
Most levels take 3 to 7 minutes of training; the two demons took 13.5 (Deadlocked) and 23 minutes (Clubstep).
Without checkpoints (every attempt from the start), the agent needs fewer attempts but 5 to 6 times more time:
26 minutes for Stereo Madness and 37 for Back on Track, because each attempt replays the whole level.
On Grief, an extreme demon, it reached 38% with the help of a backtracking search (`tools/search_stuck.py`).

![Is a level hard for the AI when it is hard for humans?](docs/figures/ai_vs_official_difficulty.png)

The levels the AI finds hardest are not always the ones humans find hardest: it never misses a timing,
but it struggles with long precise sequences and with dead ends that look promising.
All numbers, learning curves and the full ranking: **[docs/RESULTS.md](docs/RESULTS.md)**.

## How it works

- **A Geode mod** (`mod/`) drives the game in *lockstep*: the game only moves when the agent asks for a step
  (1/60 s of game time), up to 20 steps per rendered frame, so training runs up to 20x faster than real time.
- **The agent is blind.** Its state is the time since the start, its height, its vertical speed, whether it
  touches the ground and whether the button is already pressed. It never sees spikes or blocks.
- **Tabular Q-learning** with a few tricks that make it fast on a deterministic game:
  each attempt is learned backwards (reverse replay), exploration is focused on the end of the best run so far,
  exploratory actions are held for random durations (to fly a ship), and when stuck the exploration zone
  grows, then moves back to escape dead ends.
- **Practice mode:** the agent respawns at GD checkpoints placed along its best run, and every respawn is
  checked against the original run. A level only counts as beaten when the agent plays it **in one go from
  the start, without checkpoints**.

## Repository layout

```
train_qlearning.py   the agent (Q-learning) and its command line
realenv.py           the real game as a training environment (talks to the mod)
run_batch.py         trains several official levels in a row (mod v0.4.0+)
watch_best.py        replays a saved run at real speed
replay_wins.py       replays every winning run, optionally recording one video per level with OBS
make_report.py       rebuilds docs/RESULTS.md and its figures from the logs
mod/                 the Geode mod (C++), built by GitHub Actions
results/<level>/     training log and winning run of each beaten level (history/: development runs)
docs/                results report and figures
tools/               connection tests and debugging tools
sim/                 the first experiments: a simplified Python copy of GD (cube only)
```

## Getting started

```
pip install -r requirements.txt
```

**1. Build and install the mod.** Push the repository, then on GitHub open *Actions → Build Geode mod*,
open the latest run and download the `gd-ai-bridge` artifact (a `.geode` file in a zip). Copy it into
`geode/mods/` in the GD folder (Steam: right-click GD → Manage → Browse local files) and restart GD.
The mod targets GD 2.2081 and Geode 5.10 on Windows. When nothing is connected, the game behaves normally.

**2. Check the connection.** Open a level, then:
```
python -m tools.test_bridge        # running, jumping, determinism, speed
python -m tools.test_practice      # respawning at a checkpoint reproduces the run exactly
```
While Python is connected but idle, the game freezes: that is expected.

**3. Train.** Open the level in GD, then:
```
python train_qlearning.py --env real --level stereo_madness
python train_qlearning.py --env real --level stereo_madness --resume     # continue (Ctrl+C saves everything)
python run_batch.py --levels 1-9 --max-minutes 30                        # several levels in a row
```
The `--level` name is only used to name the saved files (`qtable_*.pkl`, `solution_qlearning_*.txt`).

**4. Watch, record, report.**
```
python watch_best.py solution_qlearning_stereo_madness_practice_state.txt
python replay_wins.py --obs            # one video per level in videos/ (needs OBS and obsws-python)
python make_report.py                  # after copying a finished run to results/<level>/
```

### Useful options

| Option | When to use it |
|---|---|
| `--obs fine` | Very tight passages (Clubstep, Deadlocked): 4x more precise state, more memory |
| `--seed-run FILE` | Start a new Q-table from a saved run (e.g. after changing `--obs`) |
| `--resume` | Continue from the saved Q-table |
| `--max-fall S` | Stop an attempt when the cube flies away at full speed for S seconds (default 1.5, 0 = off) |
| `--max-minutes M` | Time limit |
| `--mode normal` | No checkpoints: slower, but every attempt starts from the beginning |

### When training gets stuck

The log shows how exploration widens when no record is beaten. If it stays stuck for thousands of attempts:
```
python watch_best.py solution_qlearning_<level>_practice_state.txt   # where and how does it die?
python -m tools.diag_stuck <level>                                    # what happens around the death
python -m tools.diag_greedy <level>                                   # does the agent still follow its best run?
python -m tools.search_stuck <level>                                  # backtracking search past the wall, then --resume --seed-run
```

## First experiments: the simulator

Before connecting the real game, the idea was validated on a **simplified copy of GD in Python** (`sim/`,
cube only, levels drawn in ASCII).

![Blind agent beating the simulated level](sim/media/solution_qlearning_stereo_lite_practice_state.gif)

```
python train_qlearning.py                    # simulator, level stereo_lite
python -m sim.search_bot                     # non-AI backtracking bot, as a reference
python -m sim.compare                        # every variant over 5 seeds
```

| Variant (`stereo_lite`, 5 seeds) | Success | Steps simulated (median) |
|---|---|---|
| Search bot (reference) | yes | 1,456 |
| Blind agent, practice mode | 5/5 | ~285,000 |
| Blind agent, restarts from the beginning | 5/5 | ~1,490,000 |
| Agent that only knows x | 0/5 | — |

These numbers predate reverse replay, which made the agent several times faster.

## Next steps

- The 6 remaining official levels.
- Clean reruns of every level with the current version, several seeds each, for a fair difficulty ranking.
- Phase 2: an agent that sees the level and has to generalize to levels it has never played.
