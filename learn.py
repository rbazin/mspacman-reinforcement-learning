""" Script to train an agent on the game MsPacman with the Q-algorithm and function estimation """

import argparse
import os
import numpy as np
from random import randrange
from ale_py import ALEInterface
import learner
from time import time
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(
        description="Trains a Q-learning agent using function estimation"
    )
    parser.add_argument(
        "--bin_file",
        type=str,
        required=True,
        help="Bin file of the game MsPacman",
    )
    parser.add_argument(
        "--graph_res",
        default=False,
        action="store_true",
        help="Save the graph of the scores over all episodes of the training",
    )
    parser.add_argument(
        "--save_scores",
        default=False,
        action="store_true",
        help="Save the scores over all episodes of the training in a txt file",
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
        default=0.5,
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
        default=0.10,
        help="Sets the exploration factor of the Q-algorithm",
    )
    parser.add_argument(
        "--display_screen",
        default=False,
        action="store_true",
        dest="display_screen",
        help="Enables display of the screen, may only work on Linux",
    )
    parser.add_argument(
        "--no_food_features",
        default=True,
        action="store_false",
        dest="food_features",
        help="Disables food features in the function estimation",
    )
    parser.add_argument(
        "--no_ghost_features",
        default=True,
        action="store_false",
        dest="ghost_features",
        help="Disables ghost features in the function estimation",
    )
    return parser.parse_args()


def main():
    args = get_args()
    learner.alpha = args.alpha
    learner.gamma = args.gamma
    learner.epsilon = args.epsilon
    learner.food_features = args.food_features
    learner.ghost_features = args.ghost_features

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

    rewards = []

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

            state = learner.getState(screen_data)
            a = learner.Q_learn(state, reward)
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

        rewards.append(total_reward)

        if args.save_scores:
            with open("scores.txt", "a") as f:
                f.write(f"{episode + 1} {total_reward}\n")

        ale.reset_game()

    end = time()

    if args.graph_res:
        plt.plot(rewards, label="score")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title(f"Evolution of the score during training")
        plt.savefig("scores.png")

    print(f"Training completed in {end - start:.2f}s")

    # Save the weights
    print(f"Saving weights : {learner.theta}")
    learner.saveWeights("weights.csv")


if __name__ == "__main__":
    main()
