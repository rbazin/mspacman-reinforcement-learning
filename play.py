""" Script to make the agent play on the game MsPacman with the Q-algorithm and function estimation with pretrained weights"""

import argparse
import os
import numpy as np
from random import randrange
from ale_py import ALEInterface
import learner
from time import time


def get_args():
    parser = argparse.ArgumentParser(
        description="Make a Q-learning agent play using function estimation and pretrained weights"
    )
    parser.add_argument(
        "--bin_file",
        type=str,
        required=True,
        help="Bin file of the game MsPacman",
    )
    parser.add_argument(
        "--nbr_episodes",
        type=int,
        default=20,
        help="Sets the number of games the agent will play",
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        default=False,
        help="Enables saving the scores of the episodes played in a file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Weights of the trained model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Defines the random seed of the game",
    )
    parser.add_argument(
        "--display_screen",
        default=False,
        action="store_true",
        dest="display_screen",
        help="Enables display of the screen, may only work on Linux",
    )
    return parser.parse_args()


def main():
    args = get_args()

    ale = ALEInterface()
    ale.setInt("random_seed", args.seed)

    # Check if the bin file of the game exists
    bin_file_path = os.path.join(args.bin_file)
    assert os.path.exists(bin_file_path), "Bin for the game not found"

    # Load model weights
    path_weights = os.path.join(args.weights)
    assert os.path.exists(path_weights), "Weights not found"
    learner.theta = learner.getWeights(path_weights)
    print(f"Loading parameters : {learner.theta}")

    if args.display_screen:
        ale.setBool("display_screen", True)
        ale.setBool("sound", True)

    ale.setInt("frame_skip", 5)
    ale.loadROM(bin_file_path)

    # Get the list of minimal legal actions (some ALE actions are useless in this game)
    legal_actions = ale.getMinimalActionSet()
    num_actions = len(legal_actions)

    print("Game starting")
    for episode in range(args.nbr_episodes):
        start = time()
        total_reward = 0
        while not ale.game_over():
            (screen_width, screen_height) = ale.getScreenDims()

            screen_data = np.zeros((screen_width, screen_height), dtype=np.uint8)
            ale.getScreen(screen_data)

            state = learner.getState(screen_data)
            a, _ = learner.maxQ(state)
            if a is None:
                a = legal_actions[randrange(num_actions)]

            total_reward += ale.act(a)

        end = time()
        print(
            f"Episode {episode + 1} finished in {end - start:.2f}s with score {total_reward}"
        )

        if args.save_scores:
            with open("test_scores.txt", "a") as f:
                f.write(f"{episode + 1} {total_reward}\n")


if __name__ == "__main__":
    main()
