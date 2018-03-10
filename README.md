# Shared-WGAN

The repository is the experimental setup for the research paper: [https://arxiv.org/abs/1802.07401](https://arxiv.org/abs/1802.07401)

A research based WGAN model that has shared weights between genrator and discriminator. The premise was that since both the networks are working on the similar data, some of the transformation could beuseful. 
The experiment indicated that GANs may contain a similar structures between the networks and could be exploited in the future for imiproved GAN training.

# Setting up

 - Clone the repository
 - Run the file ```driver.py``` with the required parameters set
 - The file ```visualizer.py``` has code to convert the weights of the network into files
 - Visualize the learned weights for interesting results
 
 # Credits
 
 Code in the repository has been inspired from [https://github.com/carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
