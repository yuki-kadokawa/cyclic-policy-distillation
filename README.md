# cyclic-policy-distillation
![CPD](fig/cpd_architecture.png "Overview of CPD")

## Introduction
Deep reinforcement learning with domain randomization learns a control policy in various simulations with randomized physical and sensor model parameters to become transferable to the real world in a zero-shot setting. 
However, a huge number of samples are often required to learn an effective policy when the range of randomized parameters is extensive due to the instability of policy updates. 
To alleviate this problem, we propose a sample-efficient method named Cyclic Policy Distillation (CPD). 
CPD divides the range of randomized parameters into several small sub-domains and assigns a local policy to each sub-domain. 
Then, local policies are learned while cyclically transitioning to sub-domains. Furthermore, CPD accelerates learning through knowledge transfer according to expected performance improvements. 
Finally, all of the learned local policies are distilled into a global policy for sim-to-real transfer. 

## Video
https://youtu.be/UxRnZcLIe3c

## Installation


