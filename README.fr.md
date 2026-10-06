# Geometry Dash AI — une IA aveugle par apprentissage par renforcement

🇬🇧 [English](README.md) · 🇫🇷 Français

Une IA qui apprend à finir **les niveaux officiels de Geometry Dash, dans le vrai jeu**, par essai-erreur,
**sans jamais voir les obstacles**. Elle sait seulement où elle en est dans le niveau et comment elle bouge ;
tout le reste, elle l'apprend en mourant.

## Résultats

**15 des 22 niveaux officiels finis**, dont les deux démons tentés pour l'instant, **Clubstep** et **Deadlocked**.
La plupart des niveaux demandent 3 à 7 minutes d'entraînement ; Deadlocked en a demandé 13,5.

![Un niveau difficile pour les humains l'est-il aussi pour l'IA ?](docs/figures/ai_vs_official_difficulty.png)

Les niveaux que l'IA trouve les plus durs ne sont pas toujours ceux que les humains trouvent les plus durs :
elle ne rate jamais un timing, mais elle peine sur les longues séquences précises et sur les culs-de-sac
qui ont l'air prometteurs. Tous les chiffres, les courbes d'apprentissage et le classement complet :
**[docs/RESULTS.md](docs/RESULTS.md)**.

## Comment ça marche

- **Un mod Geode** (`mod/`) pilote le jeu *pas à pas* : le jeu n'avance que quand l'IA demande un pas
  (1/60 s de jeu), jusqu'à 20 pas par image affichée, donc l'entraînement va jusqu'à 20 fois plus vite que le temps réel.
- **L'IA est aveugle.** Son état : le temps écoulé depuis le début, sa hauteur, sa vitesse verticale, si elle
  touche le sol et si le bouton est déjà enfoncé. Elle ne voit jamais les pics ni les blocs.
- **Du Q-learning tabulaire**, avec quelques astuces qui le rendent rapide sur un jeu déterministe :
  chaque tentative est apprise à l'envers (reverse replay), l'exploration se concentre sur la fin du meilleur
  essai, les actions d'exploration sont maintenues pendant une durée aléatoire (pour piloter un vaisseau), et
  quand l'IA bloque, la zone explorée s'agrandit puis recule pour sortir des culs-de-sac.
- **Le mode practice :** l'IA réapparaît à des checkpoints de GD posés le long de son meilleur essai, et chaque
  réapparition est vérifiée par rapport à la partie d'origine. Un niveau ne compte comme fini que si l'IA le
  joue **d'une traite depuis le début, sans checkpoint**.

## Organisation du dépôt

```
train_qlearning.py   l'IA (Q-learning) et sa ligne de commande
realenv.py           le vrai jeu vu comme un environnement d'entraînement (dialogue avec le mod)
run_batch.py         entraîne plusieurs niveaux officiels à la suite (mod v0.4.0+)
watch_best.py        rejoue une partie sauvegardée à vitesse normale
replay_wins.py       rejoue toutes les victoires, avec une vidéo par niveau enregistrée par OBS si on veut
make_report.py       régénère docs/RESULTS.md et ses figures à partir des logs
mod/                 le mod Geode (C++), compilé par GitHub Actions
results/<niveau>/    log d'entraînement et partie gagnante de chaque niveau fini (history/ : essais de développement)
docs/                rapport de résultats et figures
tools/               tests de connexion et outils de débogage
sim/                 les premières expériences : une copie simplifiée de GD en Python (cube seulement)
```

## Prise en main

```
pip install -r requirements.txt
```

**1. Compiler et installer le mod.** Pousse le dépôt, puis sur GitHub ouvre *Actions → Build Geode mod*,
la dernière exécution, et télécharge l'artefact `gd-ai-bridge` (un fichier `.geode` dans un zip). Copie-le dans
`geode/mods/` du dossier de GD (Steam : clic droit sur GD → Gérer → Parcourir les fichiers locaux) et relance GD.
Le mod vise GD 2.2081 et Geode 5.10 sur Windows. Sans connexion, le jeu fonctionne normalement.

**2. Vérifier la connexion.** Ouvre un niveau, puis :
```
python -m tools.test_bridge        # course, saut, déterminisme, vitesse
python -m tools.test_practice      # réapparaître à un checkpoint reproduit exactement la partie
```
Quand Python est connecté mais ne fait rien, le jeu se fige : c'est normal.

**3. Entraîner.** Ouvre le niveau dans GD, puis :
```
python train_qlearning.py --env real --level stereo_madness
python train_qlearning.py --env real --level stereo_madness --resume     # reprendre (Ctrl+C sauvegarde tout)
python run_batch.py --levels 1-9 --max-minutes 30                        # plusieurs niveaux à la suite
```
Le nom donné à `--level` sert seulement à nommer les fichiers sauvegardés (`qtable_*.pkl`, `solution_qlearning_*.txt`).

**4. Regarder, enregistrer, faire le rapport.**
```
python watch_best.py solution_qlearning_stereo_madness_practice_state.txt
python replay_wins.py --obs            # une vidéo par niveau dans videos/ (OBS et obsws-python nécessaires)
python make_report.py                  # après avoir copié une partie finie dans results/<niveau>/
```

### Options utiles

| Option | Quand l'utiliser |
|---|---|
| `--obs fine` | Passages très serrés (Clubstep, Deadlocked) : état 4 fois plus précis, plus de mémoire |
| `--seed-run FICHIER` | Démarrer une nouvelle Q-table à partir d'une partie sauvegardée (par exemple après avoir changé `--obs`) |
| `--resume` | Reprendre à partir de la Q-table sauvegardée |
| `--max-fall S` | Arrêter un essai quand le cube s'envole à pleine vitesse pendant S secondes (1,5 par défaut, 0 = désactivé) |
| `--max-minutes M` | Limite de temps |
| `--mode normal` | Sans checkpoints : plus lent, mais chaque essai part du début |

### Quand l'entraînement bloque

Le log indique comment l'exploration s'élargit quand aucun record ne tombe. Si l'IA reste bloquée des milliers d'essais :
```
python watch_best.py solution_qlearning_<niveau>_practice_state.txt   # où et comment meurt-elle ?
python -m tools.diag_stuck <niveau>                                    # ce qui se passe autour de la mort
python -m tools.diag_greedy <niveau>                                   # suit-elle encore son meilleur essai ?
```

## Premières expériences : le simulateur

Avant de brancher le vrai jeu, l'idée a été validée sur une **copie simplifiée de GD en Python** (`sim/`,
cube seulement, niveaux dessinés en ASCII).

![IA aveugle qui finit le niveau simulé](sim/media/solution_qlearning_stereo_lite_practice_state.gif)

```
python train_qlearning.py                    # simulateur, niveau stereo_lite
python -m sim.search_bot                     # bot sans IA par retour arrière, comme référence
python -m sim.compare                        # toutes les variantes sur 5 graines
```

| Variante (`stereo_lite`, 5 graines) | Réussite | Pas simulés (médiane) |
|---|---|---|
| Bot de recherche (référence) | oui | 1 456 |
| IA aveugle, mode practice | 5/5 | ~285 000 |
| IA aveugle, recommence du début | 5/5 | ~1 490 000 |
| IA qui ne connaît que x | 0/5 | — |

Ces chiffres datent d'avant le reverse replay, qui a rendu l'IA plusieurs fois plus rapide.

## Suite

- Les 7 niveaux officiels restants.
- Relancer proprement tous les niveaux avec la version actuelle, plusieurs graines chacun, pour un classement de difficulté équitable.
- Phase 2 : une IA qui voit le niveau et doit se débrouiller sur des niveaux qu'elle n'a jamais joués.
