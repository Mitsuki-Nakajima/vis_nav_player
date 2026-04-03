# Visual Navigation Game

This is the course project for ROB-UY 3203.

# Instructions for Players
1. Install
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```
2. Include the exploration data
Download the provided exploration dataset and place it under:
```commandline
data/exploration_data/dataset/image.jpg
```
Ensure the directory structure is correct before running the program.

3.. Play using the default keyboard player
```commandline
python source/vis_nav_player.py
```
- Press "A" to toggle navigation mode AUTO <--> MANUAL
- Press "SPACE" to check-in at the goal

