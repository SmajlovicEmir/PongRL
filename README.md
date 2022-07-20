# PongRL
This is a short and simple implementation of the Classic Pong Game with an RL approach.
Just wanted to test out Stable Baselines 3, I trained the model but ran out of disk space.
The model is pretty simple, it's a classic pong game, with the twist that the agents opponent is a wall.
Make sure to remember to register the environment in stable baselines 3 so you can actually run it.
Unfortunately due to time constraints it does have a PyGame dependency, but it could also be easily made with OpenCV. 
I have also added some which you could adjust in order to train the model yourself, but do keep in mind that the parameters would need some tweaking.
As per different sources the model takes about 5.000.000 timesteps to converge at the very least.
Also, keep in mind that most of the results are hardly reproducible.
PyGame has a weird bug that if you click anywhere on the screen while the agent is learning it will crash, if the rendering is enabled, so do try to disable the rendering if you intend on doing ANYTHING else on your computer/laptop.
