# Efficient Reinforcement Learning in Probabilistic Reward Machines
## Introduction
This repository holds the implementation codes for the simulation scenarios in the paper. 

Paper Name: Efficient Reinforcement Learning in Probabilistic Reward Machines
## Getting started
**Create virtual environment**
```
conda create -n "UCBVI-PRM" python=3.8.10
conda activate UCBVI-PRM
```
**Installing packages:**
Navigate to this repo's directory
```
conda install --yes --file requirements.txt
```
## Run experiments
```
python run_exp.py
```
The script will:

-	Build the specified environment.
- Initialize the learners.
- Run the experiments across multiple episodes.
- Save the results and environment data for future analysis.

**Parameters**

	- epi_len: The length of each episode.
	- num_epi: The number of episodes per experiment.
	- num_states: The number of states in the environment. (except for warehouse environment, it will be the length of one side of warehouse, e.g. num_states = 5 will render a 5x5 warehouse)
	- num_exp: The number of independent experiments to run.
	- test_name: The name of the environment to test.

### DRM

There are three test environment for DRM:
  - river_swim_patrol
  - flower
  - two_room_2corners


There are four type of learners that can be used in DRM: UCBVI-RM, UCBVI-CP, UCRL2-RM-L and UCRL2-RM-B. To specify the learners, for example, UCBVI-RM, UCRL2-RM-L and UCRL2-RM-B, specify them in the function of **run_exp()** of file ``run_exp.py`` by creating a list of learners:
```
    learner_1 = UCBVI_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine,bonus_scale=0.001)
    learner_2 = UCRL2_RM(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, distance_scale=0.5)
    learner_3 = UCRL2_RM_Bernstein(n_states, n_actions, epi_len, delta=0.05, K=num_epi, RM=env.rewardMachine, distance_scale=0.1)
    learners.append(learner_1)
    learners.append(learner_2)
    learners.append(learner_3)

```
### PRM
There are three test environment for DRM:
  - warehouse
  - river_swim_patrol_prm

There are two type of learners that can be used in DRM: UCBVI-PRM, UCBVI-CP. To specify the learners for PRM, please apply the same method as DRM case, creating a list of learners.

### Tuning of exploration coefficient
To specify the exploration coefficient of algorithms, for UCBVI-RM, UCBVI-PRM and UCBVI-CP, specify ``bonus_scale`` for them.
For UCRL2-RM-L and UCRL2-RM-B, specify ``distance_scale`` for them.
## Hardware Environment

**Device Overview**

- Device: MacBook Air
- Chip: Apple M2
- Model: MacBook Air (M2, 2022)
- Processor: Apple M2 Chip
- Architecture: ARM-based
- RAM:16 GB
- Number of Cores: 8-core CPU 
 
  

## License
The project is licensed under MIT License.
## Acknowledgments
This project is based on the framework project [UCRL2-RM](https://github.com/HippolyteBourel/UCRL-RM). The foundation for the environment setup, baseline algorithms were inspired or adapted from their work. 