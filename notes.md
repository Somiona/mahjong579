# 想法A
suphx-reference 只做了suphx论文的第一部分，训练一个supervised learning model。根据suphx论文的介绍，他们这个系统先用SL的方法训练了四个小模型：
吃，碰，杠，立直，打牌。
![alt text](<img/CleanShot 2024-04-16 at 06.22.24.png>)
然后打牌这个模型被单独用RL训练了一轮。
![alt text](<img/CleanShot 2024-04-16 at 06.22.58.png>)

我们的工作需要完善的部分：
1. Global Reward Prediction, 根据suphx原论文的介绍，这是为了作为我们的reward function。
   ![alt text](<img/CleanShot 2024-04-16 at 06.25.18.png>)
   ![alt text](<img/CleanShot 2024-04-16 at 06.26.41.png>)

2. Oracle Guiding, 这个部分我想用 Variational Oracle Guiding来做。
    ![alt text](<img/vlog.png>)

3. Parametric Monte-Carlo Policy Adaptation 砍掉，这部分超出这节课的内容了。

# 想法B
Building a 3-Player Mahjong AI using
Deep Reinforcement Learning

这篇论文的明显要简单很多。。。RL的部分就是只有一个mcpg（我们上课学了）
![alt text](<img/CleanShot 2024-04-16 at 06.48.39.png>)

# 杂项

## multi-agent RL introduction
https://huggingface.co/learn/deep-rl-course/unit7/multi-agent-setting

## self-play RL
https://huggingface.co/learn/deep-rl-course/unit7/self-play
