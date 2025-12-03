import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CaMIL(nn.Module):
    """
    Single-head attention based Multiple Instance Learning (MIL) model
    Input: FC matrices [batch, sliding_window, ROI, ROI]
    Output: Classification probability Y_prob, binary prediction Y_hat
    """
    def __init__(self, num_windows, ROI_num=116, timepoints=30, device=None):
        super(CaMIL, self).__init__()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.ROI_num = ROI_num
        self.num_windows = num_windows
        self.timepoints = timepoints
        self.L = ROI_num * ROI_num
        self.D = 1024
        self.K = 1
        self.H = 1024
        self.Kmeans = 2

        # Convolutional feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(1, 1, 5), stride=1, padding=(0,0,2), bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(1,1,3), stride=1, padding=(0,0,1), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(32, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU()
        )


        self.weight_FC_local = nn.Parameter(torch.rand(self.timepoints))
        self.weight_FC_global = nn.Parameter(torch.rand(self.timepoints))

        # Single-head attention conv
        self.query_conv = nn.Conv3d(1, 1, kernel_size=1)
        self.key_conv = nn.Conv3d(1, 1, kernel_size=1)
        self.value_conv = nn.Conv3d(1, 1, kernel_size=1)
        self.W_z = nn.Conv3d(1,1,kernel_size=1)


        self.Wq_cross = nn.Linear(6670, self.H)
        self.Wk_cross = nn.Linear(6670, self.H)
        self.Wv_cross = nn.Linear(6670, self.H)

        self.softmax = nn.Softmax(dim=-1)

        # MIL attention layer
        self.MIL_attention = nn.Sequential(
            #nn.Linear(ROI_num*ROI_num, self.D),
            nn.Linear(6670, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.bn = nn.BatchNorm1d(6670)
        self.bn2 = nn.BatchNorm1d(6670)

        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(6670 + self.H, self.H),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(self.H, 1),
            nn.Sigmoid()
        )

    def upper_triangular(self, x):
        mask = torch.triu(torch.ones(self.ROI_num, self.ROI_num), diagonal=1).bool()
        vectors = [matrix[mask].flatten() for matrix in x]
        return torch.stack(vectors)

    def forward(self, FC, cluster_centers):
        """
        Forward pass
        FC: [batch, windows, ROI, ROI]
        cluster_centers: KMeans cluster centers
        """
        batch_size,num_windows, width, _ = FC.size()
        #bag_level = FC.view(batch_size, num_windows, width*width)
        bag_level=self.upper_triangular(FC.view(batch_size*num_windows, width,width)).view(batch_size, num_windows, -1)

        MIL_attention_list = []

        for i in range(batch_size):
            FC_flat = bag_level[i]
            attention = self.MIL_attention(FC_flat)
            attention = self.softmax(attention.transpose(1,0))
            bag_repr = torch.mm(attention, FC_flat)
            MIL_attention_list.append(bag_repr)

        #bag_z = torch.cat(MIL_attention_list, dim=0).squeeze(1).view(batch_size, width, width)
        bag_z_up = torch.cat(MIL_attention_list, dim = 0).squeeze(1)
        #bag_z_up = self.upper_triangular(bag_z)

        bag_z_up = self.bn(bag_z_up)
        cluster_centers = self.bn2(cluster_centers)
        Wq = self.Wq_cross(bag_z_up).permute(1,0)
        Wk = self.Wk_cross(cluster_centers)
        QK = torch.einsum('dB,Kd->BK', Wq, Wk) / math.sqrt(self.H)
        att_cross = self.softmax(QK)
        Wv = self.Wq_cross(cluster_centers)
        final_cross = torch.einsum('BK,Kd->Bd', att_cross, Wv)

        X_sum = torch.cat((final_cross, bag_z_up), dim=1).view(batch_size, -1)
        Y_prob = self.classifier(X_sum).squeeze(1)
        Y_hat = (Y_prob >= 0.5).float()
        return Y_prob, Y_hat

    def extract_features(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        proj_q = self.query_conv(x)
        proj_k = self.key_conv(x).permute(0,1,2,4,3)
        energy = torch.einsum('...ik,...kj->...ij', proj_q, proj_k)
        mean_energy = energy.mean(dim=2, keepdim=True)
        mean_energy_expanded = mean_energy.expand_as(energy).clone()
        energy = energy + mean_energy_expanded
        attention = self.softmax(energy)
        proj_v = self.value_conv(x)
        out = torch.einsum('...ik,...kj->...ij', attention, proj_v)
        W_y = self.W_z(out)
        z = W_y + x
        x_global_FC = torch.einsum('...ik,...kj->...ij', z, z.transpose(-2,-1))
        return x_global_FC, (energy - mean_energy).abs().mean()
    def calculate_classification_error(self, FC, Y,cluster_centers):

        Y = Y.float()
        _, Y_hat = self.forward(FC,cluster_centers)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, FC, Y, cluster_centers ):

        Y = Y.float()
        Y_prob, _ = self.forward(FC, cluster_centers)
        Y_prob = torch.clamp(Y_prob, min = 1e-5, max = 1. - 1e-5)
        neg_log_likelihood = -1. * (
                        Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return Y_prob,neg_log_likelihood.mean()
