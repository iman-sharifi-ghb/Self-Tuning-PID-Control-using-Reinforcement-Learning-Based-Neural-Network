# Self-Tuning-PID-Control-using-Reinforcement-Learning-Based-Neural-Network
Novel adaptive tuning of the PID gains using an Actor-Critic-based Neural Network for Attitude Control of a 6-Dof, 4-Motor Robot

## Abstract

Proportional-Integrator-Derivative (PID) controller is used in a wide range of industrial and experimental processes.
There are a couple of offline methods for tuning PID gains. However, due to the uncertainty of model parameters and
external disturbances, real systems such as aerial robots need more robust and reliable PID controllers. In this research, a
self-tuning PID controller using a Reinforcement-Learning-based Neural Network for attitude and altitude control of a 6-Dof, 4-motor Robot
has been investigated. An Incremental PID, which contains static and dynamic gains, has been considered and only the variable
gains have been tuned. To tune dynamic gains, a model-free actor-critic-based hybrid neural structure was used that was
able to properly tune PID gains, and also has done the best as an identifier. In both tunning and identification tasks, a Neural
Network with two hidden layers and sigmoid activation functions has been learned using an Adaptive Momentum (ADAM) optimizer
and Back-Propagation (BP) algorithm. This method is online, able to tackle disturbance, and fast in training. In addition to
robustness to mass uncertainty and wind gust disturbance, results showed that the proposed method had a better performance when
compared to a PID controller with constant gains.

You can access to the paper via this [link](https://github.com/98210184/Self-Tuniing-PID-Control-using-Reinforcement-Learning-Based-Neural-Network/blob/main/Article%20-%20Intelligent%20Self-Tuning%20PID%20Controller.pdf).

