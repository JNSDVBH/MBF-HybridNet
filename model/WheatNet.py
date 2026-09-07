import torch
import torch.nn as nn


class RemoteSensingBranch(nn.Module):
    """
    Remote sensing feature extraction branch
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )


    def forward(self,x):

        return self.encoder(x)



class ClimateBranch(nn.Module):
    """
    Meteorological and climate feature branch
    """

    def __init__(self, input_dim, hidden_dim):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )


    def forward(self,x):

        return self.encoder(x)



class FeatureFusion(nn.Module):
    """
    Multi-branch feature fusion module

    Detailed fusion strategy is described in the manuscript.
    """

    def __init__(self, feature_dim):

        super().__init__()

        self.fusion = nn.Linear(
            feature_dim,
            feature_dim
        )


    def forward(self,x):

        return self.fusion(x)



class MBF_HybridNet(nn.Module):

    def __init__(
            self,
            rs_dim,
            climate_dim,
            output_dim=1):

        super().__init__()


        self.rs_branch = RemoteSensingBranch(
            rs_dim,
            128
        )


        self.climate_branch = ClimateBranch(
            climate_dim,
            128
        )


        self.fusion = FeatureFusion(
            256
        )


        self.predictor = nn.Linear(
            256,
            output_dim
        )


    def forward(
            self,
            rs_feature,
            climate_feature):


        rs_out = self.rs_branch(
            rs_feature
        )


        climate_out = self.climate_branch(
            climate_feature
        )


        fused = torch.cat(
            [
                rs_out,
                climate_out
            ],
            dim=1
        )


        fused = self.fusion(
            fused
        )


        output = self.predictor(
            fused
        )


        return output



def build_model(
        rs_dim,
        climate_dim):

    return MBF_HybridNet(
        rs_dim,
        climate_dim
    )