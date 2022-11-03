""" Script to train an agent on the game MsPacman with the Q-algorithm and function estimation """

import argparse
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import MsPacman


def get_args():
    parser = argparse.ArgumentParser(
        description="Starts a Q-learning agent with function estimation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Defines the random seed of the game",
    )
    parser.add_argument(
        "--nbr_epochs",
        type=int,
        default=50,
        help="Defines the number of epochs to train the agent on",
    )
    return parser.parse_args()


def main():
    args = get_args()

    ale = ALEInterface()
    ale.setInt("random_seed", 123)
    ale.loadROM(MsPacman)

    # Get the list of minimal legal actions (some ALE actions are useless in this game)
    legal_actions = ale.getMinimalActionSet()
    num_actions = len(legal_actions)

    for epoch in range(args.nbr_epochs):
        # Main loop
        total_reward = 0
        while not ale.game_over():

            a = legal_actions[randrange(num_actions)]
            reward = ale.act(a)
            total_reward += reward

    ale.reset_game()
    print(f"Epoch {epoch} completed with score {total_reward}")

    # Should end with saving the weights learned


if __name__ == "__main__":
    main()
