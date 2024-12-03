import random
import torch


def select_greedy_action(state, policy_net, action_size):
    """ Select the greedy action
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select greedy action

    device =torch.device("cuda") 

    # Convert state to a PyTorch tensor and add batch dimension
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    # Compute Q-values
    q_values = policy_net(state_tensor)

    # Select greedy action
    action = torch.argmax(q_values).item()

    # find id of the action with the highest Q-value
    return action

def select_exploratory_action(state, policy_net, action_size, exploration, t):
    """ Select an action according to an epsilon-greedy exploration strategy
    Parameters
    -------
    state: np.array
        state of the environment
    policy_net: torch.nn.Module
        policy network
    action_size: int
        number of possible actions
    exploration: LinearSchedule
        linear exploration schedule
    t: int
        current time-step
    Returns
    -------
    int
        ID of selected action
    """

    # TODO: Select exploratory action
    # Determine epsilon using the exploration schedule
    epsilon = exploration.value(t)  # Get epsilon for the current time-step t
    
    # With probability epsilon, select a random action (exploration)
    if torch.rand(1).item() < epsilon:
        # Random action selection
        print("Random action selected")
        action = torch.randint(0, action_size, (1,)).item()
    else:
        # Greedy action selection
        print("Greedy action selected")
        action = select_greedy_action(state, policy_net, action_size)


    return action

def get_action_set():
    """ Get the list of available actions
    Returns
    -------
    list
        list of available actions
    """
    return [[-1.0, 0.05, 0], [1.0, 0.05, 0], [0, 0.5, 0], [0, 0, 1.0]]
