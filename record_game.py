""" Records a game of MsPacman, look at ALE github to know how to transform frames in a video """
import os
import sys
from random import randrange
from ale_py import ALEInterface
from ale_py.roms import MsPacman


def main(record_dir=os.path.join("./recorded_games/game_1"), record_game=False):
    ale = ALEInterface()
    ale.setInt("random_seed", 123)

    # Enable screen display and sound output
    ale.setBool("display_screen", True)
    ale.setBool("sound", True)

    # Specify the recording directory and the audio file path
    if record_game:
        ale.setString("record_screen_dir", record_dir)  # Set the record directory
        ale.setString("record_sound_filename", os.path.join(record_dir, "sound.wav"))

    ale.loadROM(MsPacman)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()
    num_actions = len(legal_actions)

    while not ale.game_over():
        a = legal_actions[randrange(num_actions)]
        ale.act(a)

    print(f"Finished episode. Frames can be found in {record_dir}")


if __name__ == "__main__":
    main(os.path.join("recorded_games", "game_2"))
