import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, action_size, device):
        """ Create Q-network
        Parameters
        ----------
        action_size: int
            number of actions
        device: torch.device
            device on which to the model will be allocated
        """
        super().__init__()

        self.device = device 
        self.action_size = action_size

        # TODO: Create network

        # Convolutional layers
        self.conv1 = torch.nn.Sequential (
            nn.Conv2d (4, 32, kernel_size =5, stride =2) ,
            nn.BatchNorm2d (32) ,
            nn.ReLU () ,
            nn.MaxPool2d(2) ,
            nn.Conv2d (32 , 64, kernel_size =3, stride =1) ,
            nn.BatchNorm2d (64) ,
            nn.ReLU (),
            nn.MaxPool2d(2) 
            )

        # self.conv1.to(device)
        
        # flatten the output of the convolutional layer
        self.flatten = torch.nn.Flatten()

        # self.flatten.to(device)

        # Fully connected layers
        self.conv2 = torch.nn.Sequential (
                #TODO: make it dynamic such that as the conv1 changes, it will automatically change
                nn.Linear(68096, 10000), 
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(10000, 1024), 
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(1024, action_size))

        # self.conv2.to(device)

    def forward(self, observation):
        """ Forward pass to compute Q-values
        Parameters
        ----------
        observation: np.array
            array of state(s)
        Returns
        ----------
        torch.Tensor
            Q-values  
        """

        # TODO: Forward pass through the network
        # Move the observation to the GPU
        dev=torch.device("cpu") 
        # Squeeze to remove unnecessary dimensions
        observation = torch.squeeze(observation, dim=1).to(dev)

        observation = observation.permute(0, 3, 1, 2)

        self.conv1.to(dev)

        # Put the input image through the convolutional layers
        intermediate = self.conv1(observation)

        self.flatten.to(dev)
  
        # Flatten the output of the convolutional layer
        flat = self.flatten(intermediate)

        self.conv2.to(dev)

        # Get the prediction    
        output = self.conv2(flat)

        # print(output)
        # input("Output")

        return output


