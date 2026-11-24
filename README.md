# Liar's Dice — Strategy Solver

A counterfactual regret minimization experiment for a reduced imperfect-information bidding game.

## Overview

I reduce Liar's Dice to two players with one fair three-sided die each. Each bid claims a minimum quantity of a face across both dice. The next player can raise the bid or challenge it; there are no wild faces. The challenge produces a zero-sum payoff of +1 or −1.

## Method

- Enumerate all nine private roll pairs and legal bid histories.
- Group decision points by private die and public bid history.
- Apply simultaneous CFR updates across all deals and derive a reach-weighted average strategy.
- Calculate best responses using only the information available to the acting player; observed bids change the probability weights of the opponent's die.
- Report exploitability as half the sum of the two players' best-response payoffs.

## Run

```bash
python liars_dice.py
python liars_dice.py --iterations 500
python liars_dice.py --play --seed 7
```

## Reproduced results

| Training iterations | Exploitability (units/round) |
|---:|---:|
| 1 | 0.71821 |
| 10 | 0.18026 |
| 100 | 0.03735 |

The reduction indicates that the average policy becomes harder to exploit within the defined game. The output also reports the opening-bid distribution for each private die.

## Limitations and next steps

The game is deliberately much smaller than standard Liar's Dice. The result does not transfer automatically to six-sided dice, multiple players or elimination rounds. A useful extension is a larger game with sampled regret updates and a separate evaluation benchmark.

## Requirements and files

Python 3.9+; no third-party dependencies are required.

- `liars_dice.py` — original project implementation.
- `requirements.txt` — dependency notes.
- `results/` — outputs from the example run, including [terminal output](results/example_run.txt).

The example was rerun using the bundled source. This is a research/educational project; it is not a production system.
