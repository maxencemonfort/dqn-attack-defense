# dqn-attack-defense
Implementation of Deep Q Learning to Solve Erdos-Selfridge-Spencer Games

## Article

Based on the article:

[Can Deep Reinforcement Learning Solve Erdos-Selfridge-Spencer Games?  
Maithra Raghu, Alex Irpan, Jacob Andreas, Robert Kleinberg, Quoc V. Le, Jon Kleinberg
](https://arxiv.org/pdf/1711.02301.pdf)

## How to use

  1. Install all packages in requirements.txt
  2. Install the gym environment using 
  ```
  pip install -e gym-defender/
  ```
  3. You can choose to load multiple types of defender environment : K can be 5, 10, 15 or 20 and potential can be 0.8, 0.9, 0.95, 0.97 or 0.99. Don't forget to change the name of the environment based on these values and the initial weights for the model.

## Notes

OpenAI Baselines : https://github.com/openai/baselines
Stable Baselines Guide : https://pythonawesome.com/a-fork-of-openai-baselines-implementations-of-reinforcement-learning-algorithms/
  
