import sys
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import MsPacMan


def main():
    ale = ALEInterface()
    ale.setInt("random_seed", 123)
    ale.loadROM(MsPacMan)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    num_actions = len(legal_actions)

    total_reward = 0
    # Main loop
    while not ale.game_over():
        print(legal_actions)
        a = legal_actions[randrange(num_actions)]
        # TODO: what is the reward associated to act in pacman ?
        reward = ale.act(a)
        total_reward += reward

    print(f"Episode ended with score: {total_reward}")


if __name__ == "__main__":
    main()
