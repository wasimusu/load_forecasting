from torch.utils.data import DataLoader, Dataset
from datareader import DataReader


class RegressionDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, id):
        sample = self.inputs[id], self.labels[id]
        return sample


if __name__ == '__main__':
    fname = "data/AEP_hourly.csv"
    datareader = DataReader(fname)
    X, Y = datareader.get_data()

    dataset = RegressionDataset(inputs=X, labels=Y)
    dataset_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    for i, [input, label] in enumerate(dataset_loader):
        print(input)
        print(label)
        print()
        if i == 2: break
