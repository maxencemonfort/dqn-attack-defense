# dqn-attack-defense

Implementation of Deep Q Learning to Solve Erdos-Selfridge-Spencer Games


## Article

Based on the article:

[Can Deep Reinforcement Learning Solve Erdos-Selfridge-Spencer Games?  
Maithra Raghu, Alex Irpan, Jacob Andreas, Robert Kleinberg, Quoc V. Le, Jon Kleinberg
](https://arxiv.org/pdf/1711.02301.pdf)

## How to use

  1. Install all packages in requirements.txt
  2. Install the gym environments using 
  ```
  pip install -e gym-defender/
  pip install -e gym-attacker/
  ```
  3. You can choose to load multiple types of defender environment : K can be 5, 10, 15 or 20 and potential can be 0.8, 0.9, 0.95, 0.97 or 0.99 for the defender and 1.01, 1.03, 1.05, 1.1 or 1.2 for the attacker. Don't forget to change the name of the environment based on these values and the initial weights for the model.

  You will find the main usages of our environnements and agents in [a notebook (Notebook.ipynb)](Notebook.ipynb).

## Baselines

OpenAI Baselines : https://github.com/openai/baselines

Stable Baselines Guide : https://pythonawesome.com/a-fork-of-openai-baselines-implementations-of-reinforcement-learning-algorithms/
  
## Authors

- Maxime Bourliatoux
- Maxence Monfort
