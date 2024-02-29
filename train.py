import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Subset, TensorDataset, DataLoader
import argparse
from scipy.spatial.distance import pdist, squareform
from utils import reyi_entropy
from model import MLP
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
import scipy.io as io
from sklearn.preprocessing import StandardScaler
import os



parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='The learning rate of the optimizer.')
parser.add_argument('--n_epochs', type=int, default=300, help='n_epochs =300')
args = parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

device_ids = [0]
main_device_idx = 0

c = loadmat('data/waveform.mat')

data = StandardScaler().fit_transform(c['data'])
label = np.squeeze(c['labels'])
data, label = torch.tensor(data).float(), torch.tensor(label).long()
data_set = TensorDataset(data, label)
inputs_size = data.shape[1]

kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
if __name__ == '__main__':
    best_acc = 0
    total_acc = []
    number2acc = torch.zeros(inputs_size)

    maskdata = {}

    step = -1
    for train_idxs, test_idxs in kfold.split(data, label):
        step = step +1
        train_subset = Subset(data_set, train_idxs)
        test_subset = Subset(data_set, test_idxs)
        train_loader = DataLoader(dataset=train_subset, batch_size=8, shuffle=True)
        test_loader = DataLoader(dataset=test_subset, batch_size=64, shuffle=False)

        net = MLP(inputs_size)
        cost = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=80)

        for epoch in range(args.n_epochs):
            running_loss = 0.0
            running_correct = 0
            validate_correct = 0
            testing_correct = 0
            print("Epoch {}/{}".format(epoch, args.n_epochs))
            print("-" * 10)
            net.train()
            for x, y in tqdm(train_loader):
                X1_train, y_train = x, y
                fea_z, outputs = net(X1_train)
                _, pred = torch.max(outputs.detach(), 1)
                optimizer.zero_grad()
                loss = cost(outputs, y_train)
                with torch.no_grad():
                    Z_numpy = fea_z.cpu().detach().numpy()
                    k = squareform(pdist(Z_numpy, 'euclidean'))
                    sigma = np.mean(np.mean(np.sort(k[:, :10], 1)))+1e-2
                H_Z = reyi_entropy(fea_z, sigma=sigma ** 2)
                total_loss = loss + 0.01 * H_Z
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
                running_correct += torch.sum(pred == y_train.data)
            net.eval()
            y_pred = []
            y_true = []
            for x, y in tqdm(test_loader):
                X1_test, y_test = x, y
                _, outputs = net(X1_test)
                _, pred = torch.max(outputs.data, 1)
                testing_correct += torch.sum(pred == y_test.detach())
                y_pred.extend(list(torch.argmax(outputs, dim=1).cpu().numpy()))
                y_true.extend(list(y_test.cpu().numpy()))
                Number_of_features = torch.sum(net.drop_mask_x1)
            scheduler.step()
            Test_Accuracy = torch.true_divide(testing_correct, len(test_subset))
            # print("\n", Number_of_features, Test_Accuracy)
            if epoch > 1:
                for i in range(inputs_size):
                    if Number_of_features == i and Test_Accuracy > number2acc[i]:
                        print('Saving..')
                        number2acc[i] = Test_Accuracy
            if Test_Accuracy > best_acc:
                best_acc = Test_Accuracy
        total_acc.append(list(number2acc.numpy()))

    print('saving')
    folder = 'result'
    filename = 'waveform_result.mat'
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    io.savemat(filepath, {'total_acc': total_acc})

