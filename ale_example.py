import sys
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import MsPacman


def main():
    ale = ALEInterface()
    ale.setInt("random_seed", 123)
    ale.loadROM(MsPacman)

    # Get the list of minimal legal actions (some ALE actions are useless in this game)
    legal_actions = ale.getMinimalActionSet()
    num_actions = len(legal_actions)

    total_reward = 0
    # Main loop
    while not ale.game_over():
        a = legal_actions[randrange(num_actions)]
        # TODO: what is the reward associated to act in pacman ?
        reward = ale.act(a)
        total_reward += reward

    with open("res_random_actions.txt", "a") as f:
        f.write(str(total_reward) + "\n")
    print(f"Episode ended with score: {total_reward}")


if __name__ == "__main__":
    main()