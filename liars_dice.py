"""Two-player, single-round Liar's Dice. Python 3, standard library only.

python liars_dice.py                     # train and print metrics
python liars_dice.py --play              # play against the average strategy
python liars_dice.py --iterations 500    # longer convergence experiment

Each player privately rolls a fair three-sided die. A bid claims at least
a given quantity of a face across both dice. Bids increase by quantity,
then face. No wild faces. A successful challenge earns +1, a loss earns -1.
"""

import argparse
import random

FACES = (1, 2, 3)
BIDS = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
CALL = -1
DEALS = [(first, second) for first in FACES for second in FACES]


def actions(history):
    if not history:
        return list(range(len(BIDS)))
    last_bid = history[-1]
    return [CALL] + list(range(last_bid + 1, len(BIDS)))


def call_reward(dice, last_bid):
    """Return the challenger's payoff."""
    quantity, face = BIDS[last_bid]
    if dice.count(face) >= quantity:
        return -1
    return 1


def normalize(weights):
    total = sum(weights)
    if total == 0:
        return [1 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def train_step(regrets, totals):
    """One simultaneous CFR update over every roll and legal history."""
    # Use the same strategy for all nine rolls in this training step.
    strategies = {}

    def visit(dice, history, reach):
        player = len(history) % 2
        key = (dice[player], history)  # Everything this player can see.
        moves = actions(history)
        if key not in regrets:
            regrets[key] = [0.0] * len(moves)
            totals[key] = [0.0] * len(moves)
        if key not in strategies:
            positive_regrets = [max(0, regret) for regret in regrets[key]]
            strategies[key] = normalize(positive_regrets)

        strategy = strategies[key]
        values = []
        expected = 0.0
        for move, probability in zip(moves, strategy):
            if move == CALL:
                value = call_reward(dice, history[-1])
            else:
                next_reach = list(reach)
                next_reach[player] *= probability
                value = -visit(dice, history + (move,), next_reach)
            values.append(value)
            expected += probability * value

        for i, value in enumerate(values):
            # How much better would this move have been than our usual mix?
            regret = value - expected
            regrets[key][i] += reach[1 - player] * regret
            totals[key][i] += reach[player] * strategy[i]
        return expected

    for dice in DEALS:
        visit(dice, (), (1.0, 1.0))


def best_response(policy, seat):
    """Maximize payoff given only our die and the observed public bids."""
    def visit(own, history, weights):
        our_turn = len(history) % 2 == seat
        moves = actions(history)
        values = []
        for i, move in enumerate(moves):
            branch = weights
            if not our_turn:
                # An observed bid changes how likely each opposing die is.
                branch = []
                for other, weight in zip(FACES, weights):
                    probability = policy[other, history][i]
                    branch.append(weight * probability)

            if move == CALL:
                value = 0.0
                for other, weight in zip(FACES, branch):
                    value += weight * call_reward((own, other), history[-1])
                if not our_turn:
                    value = -value
            else:
                value = visit(own, history + (move,), branch)
            values.append(value)
        # Select one response across all still-possible opposing dice.
        if our_turn:
            return max(values)
        return sum(values)

    starting_weights = [1 / len(DEALS)] * len(FACES)
    return sum(visit(own, (), starting_weights) for own in FACES)


def label(move):
    if move == CALL:
        return "call liar"
    quantity, face = BIDS[move]
    return f"at least {quantity} showing {face}"


def ask_player(moves):
    for number, move in enumerate(moves, 1):
        print(f"{number}. {label(move)}")

    while True:
        try:
            choice = int(input("Your choice: "))
        except ValueError:
            choice = 0
        if 1 <= choice <= len(moves):
            return moves[choice - 1]
        print("Enter a listed number.")


def play(policy, rng):
    human = rng.randrange(2)
    dice = rng.choice(DEALS)
    print(f"You are player {human + 1}. Your die shows {dice[human]}.")
    history = ()
    while True:
        player = len(history) % 2
        moves = actions(history)
        if player == human:
            move = ask_player(moves)
        else:
            move = rng.choices(moves, weights=policy[dice[player], history])[0]
            print("Bot:", label(move))
        if move == CALL:
            winner = player if call_reward(dice, history[-1]) == 1 else 1 - player
            print(f"Dice (P1, P2): {dice}.", "You win!" if winner == human else "Bot wins.")
            return
        history += (move,)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--seed", type=int, help="reproduce deals and bot moves in play mode")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    regrets, totals = {}, {}
    print("Iterations   P1 best response   P2 best response   Exploitability")
    for step in range(1, args.iterations + 1):
        train_step(regrets, totals)
        if step in (1, 10, 100, 1000, args.iterations):
            policy = {key: normalize(row) for key, row in totals.items()}
            first = best_response(policy, 0)
            second = best_response(policy, 1)
            exploitability = (first + second) / 2
            print(f"{step:10} {first:18.5f} {second:18.5f} {exploitability:16.5f}")
    print("\nOpening bids by private die (average strategy):")
    for face in FACES:
        openings = []
        for bid, probability in zip(BIDS, policy[face, ()]):
            openings.append(f"{bid} {probability:.1%}")
        print(f"Die {face}:", " | ".join(openings))
    if args.play:
        play(policy, random.Random(args.seed))


if __name__ == "__main__":
    main()
