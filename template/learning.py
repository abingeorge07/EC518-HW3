import numpy as np
import torch
import torch.nn.functional as F


def perform_qlearning_step(policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device):
    """ Perform a deep Q-learning step
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    optimizer: torch.optim.Adam
        optimizer
    replay_buffer: ReplayBuffer
        replay memory storing transitions
    batch_size: int
        size of batch to sample from replay memory 
    gamma: float
        discount factor used in Q-learning update
    device: torch.device
        device on which to the models are allocated
    Returns
    -------
    float
        loss value for current learning step
    """

    # TODO: Run single Q-learning step
    """ Steps: 
        1. Sample transitions from replay_buffer
        2. Compute Q(s_t, a)
        3. Compute \max_a Q(s_{t+1}, a) for all next states.
        4. Mask next state values where episodes have terminated
        5. Compute the target
        6. Compute the loss
        7. Calculate the gradients
        8. Clip the gradients
        9. Optimize the model
    """

    # Check if replay buffer has enough samples for a batch
    if len(replay_buffer) < batch_size:
        return None  # Not enough samples to train yet

    # Sample a batch of transitions from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # 2. Compute Q(s_t, a)

    # Move states to GPU
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)

    # Compute Q(s_t, a)
    q_values = policy_net(states)

    # Compute the target Q value
    with torch.no_grad():
        # 3. Compute \max_a Q(s_{t+1}, a) for all next states.
        next_q_values = target_net(next_states)
        next_q_values = next_q_values.max(dim=1)[0]

    # 4. Mask next state values where episodes have terminated
    next_q_values[dones] = 0

    # 5. Compute the target
    target_q_values = rewards + gamma * next_q_values


    # 6. Compute the loss
    loss = F.mse_loss(q_values.gather(1, actions.unsqueeze(-1)), target_q_values.unsqueeze(-1))

    # Perform gradient descent
    optimizer.zero_grad()
    loss.backward()

    # Step 8: Clip the gradients (optional, to prevent exploding gradients)
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

    # Step 9: Optimize the model
    optimizer.step()

    return loss.item()

def update_target_net(policy_net, target_net):
    """ Update the target network
    Parameters
    -------
    policy_net: torch.nn.Module
        policy Q-network
    target_net: torch.nn.Module
        target Q-network
    """

    # TODO: Update target network
    target_net.load_state_dict(policy_net.state_dict())


