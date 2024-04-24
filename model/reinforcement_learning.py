
from mahjong.agent import AiAgent


def monte_carlo_policy_gradient(agent: AiAgent, game_data, gamma=0.9, step_size=0.01):
    # this function is being called once for each game round. Can be considered as a full episode.
    # data = [state, action, reward]
    # state : oracle state, this should be the same dimension as the input of the model
    # action : the discard of this agent, number representing 打了什么牌, range in [0, 33]
    # reward : the reward of this action, float, negative or positive
    # example data:
    #    [array([[0., 0., 0., ..., 0., 1., 1.],
    #    [0., 0., 0., ..., 0., 0., 0.],
    #    [0., 0., 0., ..., 0., 0., 0.],
    #    ...,
    #    [0., 0., 0., ..., 0., 0., 0.],
    #    [0., 0., 0., ..., 0., 0., 0.],
    #    [0., 0., 0., ..., 0., 0., 0.]]), 8, 115.32092094421387]
    print(game_data[0])
    # we want to train agent.discard_model
    # class DiscardModel(nn.Module):
    # def __init__(self, in_channels, num_layers=20):
    #     super(DiscardModel, self).__init__()
    #     self.in_conv = nn.Sequential(
    #         nn.Conv1d(in_channels, 256, kernel_size=3, padding='same'),
    #         nn.BatchNorm1d(256),
    #         nn.LeakyReLU(0.2)
    #     )
    #     self.res_blocks = nn.Sequential(
    #         *(ResBlock() for _ in range(num_layers))
    #     )
    #     self.out_conv = nn.Sequential(
    #         nn.Conv1d(256, 1, kernel_size=1),
    #         nn.BatchNorm1d(1)
    #     )

    # def forward(self, x):
    #     x = self.in_conv(x)
    #     x = self.res_blocks(x)
    #     x = self.out_conv(x)
    #     return x.squeeze(1)


