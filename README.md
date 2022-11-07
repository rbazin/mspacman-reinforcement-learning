# PacMan Reinforcement Learning Project

AI agent learning how to play the game MsPacMan (Atari 2600) thanks to basic Reinforcement Learning methods such as Q-learning algorithm. This project was made for the course ECSE 526 at McGill University.

## Install requirements

The version of python used in this project was Python3.9

You can install all the required libraries with the simple command :

```bash
pip install -r ./requirements.txt
```

## Train the model

To train the model with default parameters, use the following command :

```python
python learn.py --bin_file ./roms/mspacman.bin
```

To train the model using personalized parameters, you can use different flags :

```python
python learn.py --bin_file ./roms/mspacman.bin --alpha 0.3 --epsilon 0.90 --nbr_episodes 100
```

All the personalizable parameters can be viewed using the -h flag :
```python
python learn.py -h
```

## Test the model

You can start a gain with this model using the following command :
```python
python play.py --bin_file ./roms/mspacman.bin --weights weights.csv
```

You can enable the video and audio display of the game with the --audio and --video flags (might only work on Linux) :

```python
python play.py --bin_file ./roms/mspacman.bin --weights weights.csv --display_screen
```

Once again, to display help use the -h flag :

```python
python play.py -h
```

Additionnally, you can test the model multiple times in a row with only one command using the bash script :

```bash
bash play_multiple_times.sh 10
```