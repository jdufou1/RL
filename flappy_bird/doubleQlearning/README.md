# Double Q Learning for Flappy Bird environment

I discritized the environment space by using the position of the bird and the speed because Qlearning is a tabular 
algorithm. So, with this version you need a lot of memory space to store a Q-table : (334 x 280 x 22) : states x (2) : actions => 
4 114 880 float values are saved. This is why I didnt saved the table on this repo. To reproduce the results just download the repo and 
run the **mainFlappyBird.py** file. After 8/9 hours, you will have to remove some comments in the file. The bird is capable to reach an average score of 50.

<p align="center">
  <img src="https://github.com/jdufou1/RL/blob/main/img/flappy_bird.gif" alt="animated" height="450px"/>
</p>
