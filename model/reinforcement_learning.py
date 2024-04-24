import torch
import torch.optim as optim
import torch.nn.functional as F
from mahjong.agent import AiAgent
from model.models import ResBlock


def monte_carlo_policy_gradient(agent: AiAgent, game_data, gamma=0.9, step_size=0.01):
    states, actions, rewards = zip(*game_data)

    # Convert states, actions, and rewards to PyTorch tensors
    states_tensor = torch.tensor(states, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.int64)
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

    # We want to train agent.discard_model, create optimizer
    optimizer = optim.Adam(agent.discard_model.parameters(), lr=step_size)

    G = 0
    policy_gradient = []

    # Calculate the return, going backwards from the end of the episode
    for reward in rewards[::-1]:
        G = reward + gamma * G
        policy_gradient.insert(0, G)

    policy_gradient_tensor = torch.tensor(policy_gradient, dtype=torch.float32)

    optimizer.zero_grad()

    # Forward pass
    logits = agent.discard_model(states_tensor).squeeze(-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log probabilities of the actions taken
    log_probs_actions = log_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(-1)

    # Calculate policy gradient loss
    loss = -torch.sum(log_probs_actions * policy_gradient_tensor)

    # Backward pass and optimize
    loss.backward()
    optimizer.step()

    return loss.item()  # Return the loss value as feedback
