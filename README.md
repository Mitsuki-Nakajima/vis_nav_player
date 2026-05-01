# Visual Navigation Game

This is the course project for ROB-UY 3203.

# Instructions for Players
## 1. Install
```commandline
conda update conda
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
```
## 2. Include the exploration data
Download the provided exploration dataset and place it under:
```commandline
data/exploration_data/dataset/image.jpg
```
Ensure the directory structure is correct before running the program.

## 3. Play using the default keyboard player
```commandline
python source/vis_nav_player.py
```
- Press "A" to toggle navigation mode AUTO <--> MANUAL
- Press "SPACE" to check-in at the goal

---
# Visual Navigation in Maze (Robot Vision Project)

This project implements a vision-based navigation system for an agent operating in a maze environment using only first-person view (FPV) images and a target goal image.

The system performs visual localization, path planning, and motion control without access to GPS, LiDAR, or pre-built maps. It was developed as part of a Robot Vision course and improved through midterm and final challenge iterations.


## Overview

The goal is to navigate from an unknown starting position to a target location using only visual input.

Key challenges include:
- localization under viewpoint changes
- visually repetitive environments
- control instability (oscillation, getting stuck)
- accurate goal detection


## Method

Our final system combines **visual localization, graph-based planning, motion feedback, and semi-autonomous control**.

### 1. Visual Localization (VLAD)

We use VLAD (Vector of Locally Aggregated Descriptors) for place recognition:

- Extract features from the current FPV image
- Compare against a database of reference images
- Select the most likely location (node) based on similarity

To improve stability:
- apply **temporal smoothing** over recent matches  
- reduce sudden incorrect localization jumps  

### 2. Graph-Based Path Planning

The environment is represented as a **trajectory graph**:

- nodes → reference images (locations)
- edges → feasible transitions

Navigation:
- estimate current node
- identify goal node
- compute shortest path (BFS / graph search)
- convert path into actions (forward / turn)


### 3. Motion Feedback (KLT Optical Flow)

We introduce motion verification using:

- **Shi-Tomasi corner detection**
- **Lucas-Kanade optical flow (KLT)**

This allows the system to:

- verify whether actions result in actual movement  
- detect failure cases:
  - stuck against wall  
  - oscillating (left-right loops)  
- trigger recovery behaviors when no progress is detected  


### 4. Control Strategy (Closed-Loop System)

The controller integrates:

- planner output  
- localization confidence  
- motion feedback  

Behavior:
- follow planned path when reliable  
- override actions when:
  - motion failure is detected  
  - localization confidence is low  
- perform recovery:
  - backward motion  
  - reorientation  

This forms a **closed-loop navigation system**:
> decision → action → verify → adjust


### 5. Semi-Autonomous Navigation

To improve robustness, we adopt a **semi-autonomous approach**:

- system handles navigation automatically  
- user can override controls when needed  

This helps mitigate:
- localization noise  
- accumulated errors  
- uncertain goal detection  


### 6. Goal Detection

Goal detection is based on **visual similarity** between:

- current FPV frame  
- goal image  

This operates independently from graph localization and improves final positioning accuracy.



## Improvements from Midterm

Compared to the midterm version, we introduced:

- temporal smoothing for stable localization  
- motion feedback (KLT) for detecting lack of progress  
- improved recovery behaviors (dead-ends, wall collisions)  
- reduced oscillatory control behavior  
- semi-autonomous strategy for higher reliability  

We also explored YOLO-based object detection, but found it less robust in this environment.



## Controls

- `A` → toggle AUTO / MANUAL mode  
- Arrow keys → manual control  
- `SPACE` → check-in at goal  


## Results

The final system demonstrates:

- more stable localization  
- reduced oscillation and stuck behavior  
- improved ability to recover from failure cases  
- more consistent navigation in larger maze environments  
