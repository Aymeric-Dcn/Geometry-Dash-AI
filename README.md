# Geometry Dash AI — Step 0: mini simulator + blind agent

🇬🇧 [English](#english) · 🇫🇷 [Français](#français)

![Blind agent beating the level](solution_qlearning_stereo_lite_practice_state.gif)

---

## English

Before hooking up the real game, the idea is validated on a **simplified Python copy of GD** (cube mode only).
The agent is **blind**: it sees no obstacle at all and learns by trial and error
("here, jumping killed me / didn't kill me").

### Installation

```
pip install -r requirements.txt
```

### Files

| File | Purpose |
|---|---|
| `gdsim.py` | Cube physics (240 Hz, deterministic) and training environment (Gymnasium-like API) |
| `levels.py` | Levels drawn in ASCII (`^` spike, `#` block). Add your own here |
| `search_bot.py` | Non-AI backtracking bot: proves a level is beatable, used as a reference |
| `train_qlearning.py` | The blind agent (Q-learning), with or without practice mode |
| `render.py` | Turns a solution into a GIF |
| `compare.py` | Compares every variant over 5 seeds |
| `plot_curves.py` | Plots the learning curves |

### Commands

```
python train_qlearning.py                     # blind agent, practice mode, stereo_lite level
python train_qlearning.py --mode normal       # always restarts from the beginning
python train_qlearning.py --obs x             # only knows its x position
python render.py solution_qlearning_stereo_lite_practice_state.txt
python compare.py
python plot_curves.py
```

### First results (`stereo_lite` level, 5 seeds)

| Variant | Success | Steps simulated (median) | Real-time equivalent |
|---|---|---|---|
| Search bot (reference) | yes | 1,456 | < 1 min |
| Blind agent, practice mode | 5/5 | ~285,000 | ~1 h 20 |
| Blind agent, restarts from the beginning | 5/5 | ~1,490,000 | ~7 h |
| Agent that only knows x | 0/5 | — | — |

![Learning curves](curves_stereo_lite.png)

Takeaways:
1. **The blind agent works**, as long as it knows its own movement (height, vertical speed, grounded or not).
   With x alone, it mixes up different situations (in the air / on the ground) and fails.
2. **Practice mode cuts training time by ~5x.** Keep it for the real game.
3. **A speedhack will be essential**: in real time, even a very easy level would take hours.
4. A simple search bot is hundreds of times more efficient on this problem. RL becomes worthwhile
   in phase 2 (an agent that sees the level and must generalize to unseen levels).

### Next steps

- Add the ship (hold the button) to the simulator.
- Hook up the real game through a Geode mod: read x, y, speed, death; send inputs; speedhack; respawn.

---

## Français

Avant de brancher le vrai jeu, on valide l'idée sur une **copie simplifiée de GD en Python** (mode cube uniquement).
L'IA est **aveugle** : elle ne voit aucun obstacle et apprend par essai-erreur
(« à cet endroit, sauter m'a tué / ne m'a pas tué »).

### Installation

```
pip install -r requirements.txt
```

### Fichiers

| Fichier | Rôle |
|---|---|
| `gdsim.py` | Physique du cube (240 Hz, déterministe) et environnement d'entraînement (API style Gymnasium) |
| `levels.py` | Les niveaux, dessinés en ASCII (`^` pic, `#` bloc). Ajoute les tiens ici |
| `search_bot.py` | Bot sans IA par retour arrière : prouve qu'un niveau est faisable, sert de référence |
| `train_qlearning.py` | L'IA aveugle (Q-learning), avec ou sans mode practice |
| `render.py` | Transforme une solution en GIF |
| `compare.py` | Compare toutes les variantes sur 5 graines |
| `plot_curves.py` | Trace les courbes d'apprentissage |

### Commandes

```
python train_qlearning.py                     # IA aveugle, mode practice, niveau stereo_lite
python train_qlearning.py --mode normal       # recommence toujours du début
python train_qlearning.py --obs x             # ne connaît que sa position x
python render.py solution_qlearning_stereo_lite_practice_state.txt
python compare.py
python plot_curves.py
```

### Premiers résultats (niveau `stereo_lite`, 5 graines)

| Variante | Réussite | Pas simulés (médiane) | Équivalent en temps réel |
|---|---|---|---|
| Bot recherche (référence) | oui | 1 456 | < 1 min |
| IA aveugle, mode practice | 5/5 | ~285 000 | ~1 h 20 |
| IA aveugle, recommence du début | 5/5 | ~1 490 000 | ~7 h |
| IA ne connaissant que x | 0/5 | — | — |

Ce qu'on en retire :
1. **L'IA aveugle marche**, à condition qu'elle connaisse son propre mouvement (hauteur, vitesse, au sol ou non).
   Avec seulement x, elle mélange des situations différentes (en l'air / au sol) et n'y arrive pas.
2. **Le mode practice divise le temps par ~5.** À garder pour le vrai jeu.
3. **Le speedhack sera indispensable** : en temps réel, il faudrait des heures même pour un niveau très facile.
4. Un simple bot de recherche est des centaines de fois plus efficace sur ce problème. Le RL deviendra
   intéressant en phase 2 (une IA qui voit le niveau et doit généraliser à des niveaux inconnus).

### Suite

- Ajouter le vaisseau (maintenir la touche) dans le simulateur.
- Brancher le vrai jeu via un mod Geode : lire x, y, vitesse, mort ; envoyer les inputs ; speedhack ; respawn.
