import torch
import torch.nn as nn

device = torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, round(input_size*5.5))
        self.relu = nn.ReLU()
        self.fc_add = nn.Linear(round(input_size*5.5), round(input_size*4.2))
        self.fc2 = nn.Linear(round(input_size*4.2), round(input_size*2.6))

        self.classifier = nn.Linear(round(input_size*2.6), 3)
        self.tau_x1 = torch.nn.Parameter(torch.tensor(1.))
        self.fea_dp_b_x1 = torch.nn.Parameter(torch.rand(1, input_size))
        self.drop_mask_x1_hard = 0
        # self.fea_dp_x1 = 0
    def get_retain_mask(self, drop_probs, shape, tau):
        uni = torch.rand(shape).to(device)
        eps = torch.tensor(1e-8).to(device)
        tem = (torch.log(drop_probs + eps) - torch.log(1 - drop_probs + eps) + torch.log(uni + eps) - torch.log(
            1.0 - uni + eps))
        mask = 1.0 - torch.sigmoid(tem / tau)
        return mask

    def forward(self, x, hard=False):
        self.fea_dp_b_x1_e = -3 * self.fea_dp_b_x1 + 1
        self.fea_dp_x1 = torch.sigmoid(self.fea_dp_b_x1_e)
        if hard:
            p = 0.05
            self.drop_mask_x1_hard = (self.fea_dp_x1 < p).float()
            self.drop_mask_x1 = (self.drop_mask_x1_hard - self.fea_dp_x1).detach() + self.fea_dp_x1
            self.fea_d_scale_x1 = self.fea_dp_x1.shape[0] / (torch.sum(self.drop_mask_x1) + 1e-5)
        else:
            p = 0.16
            self.drop_mask_x1 = self.get_retain_mask(drop_probs=self.fea_dp_x1, shape=self.fea_dp_x1.shape,
                                                     tau=self.tau_x1)
            self.drop_mask_x1_hard = (self.drop_mask_x1 < p).float()
            self.drop_mask_x1 = (self.drop_mask_x1_hard - self.drop_mask_x1).detach() + self.drop_mask_x1
            self.fea_d_scale_x1 = self.fea_dp_x1.shape[0] / (torch.sum(self.drop_mask_x1) + 1e-5)

        self.z_mask_x = x * self.drop_mask_x1 * self.fea_d_scale_x1
        out1 = self.fc1(self.z_mask_x)
        out2 = self.relu(out1)
        out2 = self.fc_add(out2)
        out2 = self.relu(out2)
        out2 = self.fc2(out2)
        out = self.classifier(out2)
        return self.z_mask_x, out