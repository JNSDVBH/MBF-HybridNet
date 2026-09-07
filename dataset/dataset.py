import torch
from torch.utils.data import Dataset

Batch_size = 6


def split_data_leave_one_out(
        feature_list,
        target_list,
        auxiliary_list,
        climate_list):
    """
    Leave-one-group-out validation split.

    Args:
        feature_list:
            input feature samples

        target_list:
            yield labels

        auxiliary_list:
            auxiliary information

        climate_list:
            additional climate features
    """

    indices = list(range(len(feature_list)))

    num_groups = 16

    groups = [
        indices[i::num_groups]
        for i in range(num_groups)
    ]

    for test_group in range(num_groups):
        test_indices = groups[test_group]

        train_indices = [
            idx
            for i, g in enumerate(groups)
            if i != test_group
            for idx in g
        ]

        train_x = [
            feature_list[i]
            for i in train_indices
        ]

        test_x = [
            feature_list[i]
            for i in test_indices
        ]

        train_y = [
            target_list[i]
            for i in train_indices
        ]

        test_y = [
            target_list[i]
            for i in test_indices
        ]

        yield (
            train_x,
            test_x,
            train_y,
            test_y,
            test_group
        )


class train_dataset(Dataset):

    def __init__(
            self,
            data,
            labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index]),
            torch.tensor(self.labels[index])
        )

    def __len__(self):
        return len(self.data)


class test_dataset(Dataset):

    def __init__(
            self,
            data,
            labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index]),
            torch.tensor(self.labels[index])
        )

    def __len__(self):
        return len(self.data)