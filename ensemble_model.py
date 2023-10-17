import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadNet(nn.Module):
    def __init__(self, head_hidden_size, ip_size, n_pseudo_rewards):
        super(HeadNet, self).__init__()
        self.fc1 = nn.Linear(ip_size, head_hidden_size)
        self.fc2 = nn.Linear(head_hidden_size, n_pseudo_rewards)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CoreHistoryConditionedRecurrentFeatureNetwork(nn.Module):

    def __init__(self, obs_space, output_embedding_size,
                 num_actions, action_embedding_size=32, pseudo_reward_embedding_size=32,
                 observation_embedding_size=64, lstm_output_size=128,
                 pano_obs=False):

        super().__init__()

        # Define image embedding
        self.recurrent = True
        self.number_of_actions = num_actions
        self.action_embedding_size = action_embedding_size
        self.pseudo_reward_embedding_size = pseudo_reward_embedding_size
        self.observation_embedding_size = observation_embedding_size
        self.lstm_output_size = lstm_output_size
        self.output_embedding_size = output_embedding_size
        self.pano_obs = pano_obs

        if self.pano_obs:
            self.image_conv = nn.Sequential(
                nn.Conv2d(12, 32, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(64, 128, (2, 2)),
                nn.ReLU()
            )
        else:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 32, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(32, 64, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(64, 128, (2, 2)),
                nn.ReLU()
            )
        n = obs_space["image"][0]
        m = obs_space["image"][1]

        self.flattened_size = ((n-1)//2-2)*((m-1)//2-2)*128

        self.action_embedding = nn.Linear(self.number_of_actions,
                                          self.action_embedding_size)

        self.pseudo_reward_embedding = nn.Linear(self.output_embedding_size,
                                                 self.pseudo_reward_embedding_size)

        self.lstm_input_embedding = nn.Linear(self.flattened_size,
                                              self.observation_embedding_size)

        self.memory_rnn = nn.LSTMCell(self.observation_embedding_size + self.action_embedding_size + self.pseudo_reward_embedding_size,
                                      self.lstm_output_size)
    @property
    def memory_size(self):
        return 2*self.lstm_output_size

    @property
    def semi_memory_size(self):
        return self.lstm_output_size

    def forward(self, obs, prev_action, prev_cumulant, memory):

        if self.pano_obs:
            x = obs.panorama_image.transpose(1, 3).transpose(2, 3)
        else:
            x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        x = self.lstm_input_embedding(x)
        x_a = self.action_embedding(prev_action)
        x_c = self.pseudo_reward_embedding(prev_cumulant)

        x = torch.cat((x, x_a, x_c), axis=-1)

        # separate combined h and c
        hidden = (memory[:, :self.semi_memory_size],
                  memory[:, self.semi_memory_size:])
        hidden = self.memory_rnn(x, hidden)
        embedding = hidden[0]
        memory = torch.cat(hidden, dim=1)

        return embedding, memory


class EnsembledHistoryConditionedRecurrentFeatureNetwork(nn.Module):

    def __init__(self, obs_space, output_embedding_size, num_actions,
                 action_embedding_size=32, pseudo_reward_embedding_size=32,
                 observation_embedding_size=64, lstm_output_size=128,
                 head_hidden_size=256, n_ensemble=2, pano_obs=False):

        super().__init__()
        self.recurrent = True
        self.num_heads = n_ensemble
        self.core_net = CoreHistoryConditionedRecurrentFeatureNetwork(
                                                    obs_space,
                                                    output_embedding_size,
                                                    num_actions,
                                                    action_embedding_size,
                                                    pseudo_reward_embedding_size,
                                                    observation_embedding_size,
                                                    lstm_output_size,
                                                    pano_obs=pano_obs)

        self.net_list = nn.ModuleList([HeadNet(head_hidden_size=head_hidden_size, ip_size=lstm_output_size, n_pseudo_rewards=output_embedding_size) for k in range(n_ensemble)])

    @property
    def memory_size(self):
        return 2*self.core_net.lstm_output_size

    @property
    def semi_memory_size(self):
        return self.core_net.lstm_output_size

    def _core(self, obs, prev_action, prev_cumulant, memory):
        return self.core_net(obs, prev_action, prev_cumulant, memory)

    def _heads(self, x):
        return [net(x) for net in self.net_list]

    def forward(self, obs, prev_action, prev_cumulant, memory):
        core_embedding, memory = self._core(obs, prev_action, prev_cumulant,
                                            memory)
        net_heads = self._heads(core_embedding)
        return net_heads, memory
