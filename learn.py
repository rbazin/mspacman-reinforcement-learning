""" Script to train an agent on the game MsPacman with the Q-algorithm and function estimation """

import argparse
import os
import numpy as np
from random import randrange
from ale_py import ALEInterface
import RL
from time import time


def get_args():
    parser = argparse.ArgumentParser(
        description="Starts a Q-learning agent with function estimation"
    )
    parser.add_argument(
        "--bin_file",
        type=str,
        required=True,
        help="Bin file of the game MsPacman",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Defines the random seed of the game",
    )
    parser.add_argument(
        "--nbr_episodes",
        type=int,
        default=50,
        help="Defines the number of episodes to train the agent on",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Sets the learning rate of the Q-algorithm",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Sets the discount factore of the Q-algorithm",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Sets the exploration factor of the Q-algorithm",
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
    RL.alpha = args.alpha
    RL.gamma = args.gamma
    RL.epsilon = args.epsilon

    ale = ALEInterface()
    ale.setInt("random_seed", args.seed)

    # Check if the bin file of the game exists
    bin_file_path = os.path.join(args.bin_file)
    assert os.path.exists(bin_file_path), "Bin for the game not found"

    if args.display_screen:
        ale.setBool("display_screen", True)
        ale.setBool("sound", True)

    ale.setInt("frame_skip", 5)
    ale.loadROM(bin_file_path)

    # Get the list of minimal legal actions (some ALE actions are useless in this game)
    legal_actions = ale.getMinimalActionSet()
    num_actions = len(legal_actions)

    start = time()
    for episode in range(args.nbr_episodes):
        # Main loop
        reward = 0
        total_reward = 0
        start_episode = time()
        while not ale.game_over():
            (screen_width, screen_height) = ale.getScreenDims()

            screen_data = np.zeros((screen_width, screen_height), dtype=np.uint8)
            ale.getScreen(screen_data)

            state = RL.getState(screen_data)
            a = RL.Q_learn(state, reward)
            if a is None:
                a = legal_actions[randrange(num_actions)]

            # Apply an action and get the resulting reward
            reward = ale.act(a)
            total_reward += reward

        # End of episode
        end_episode = time()
        print(
            f"Episode {episode + 1} completed with score {total_reward} in {end_episode - start_episode:.2f}s"
        )

        with open("scores.txt", "a") as f:
            f.write(f"{episode + 1} {total_reward}\n")

        ale.reset_game()

    end = time()

    print(f"Training completed in {end - start:.2f}s")
    # Should end with saving the weights learned
    RL.saveWeights("weights.csv")


if __name__ == "__main__":
    main()
