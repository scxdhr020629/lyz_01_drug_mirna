import logging
import sys

import pandas as pd
import torch.nn as nn
from jinja2.lexer import TOKEN_DOT
from torch.utils.data import DataLoader, WeightedRandomSampler
from model.cnn_gcnmulti import GCNNetmuti
# from  model.cnn_gcn import GCNNet
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

# training function at each epoch

drug1 = ""



def train(model, device, train_loader, optimizer, epoch):
    logging.info('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)).requires_grad_(True)
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            log_msg = 'Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data.x),
                                                                               len(train_loader.dataset),
                                                                               100. * batch_idx / len(train_loader),
                                                                               loss.item())
            logging.info(log_msg)


 #        for name, param in model.named_parameters():
 #            if param.grad is not None:
 #                print(f"Gradient for {name} - min: {param.grad.min()}, max: {param.grad.max()}")
 #            else:
 #                print(f"No gradient for {name}")

def predicting(model, device, loader):
    model.eval()
    total_probs = []
    sample_indices = []
    total_labels = []

    logging.info('Making predictions for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):

            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy()
            indices = np.arange(len(probs)) + batch_idx * loader.batch_size

            total_probs.extend(probs)
            sample_indices.extend(indices)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    sample_indices = np.array(sample_indices).flatten()
    total_labels = np.array(total_labels).flatten()

    accuracy = accuracy_score(total_labels, (total_probs >= 0.5).astype(int))
    precision = precision_score(total_labels, (total_probs >= 0.5).astype(int), zero_division=1)
    recall = recall_score(total_labels, (total_probs >= 0.5).astype(int))
    f1 = f1_score(total_labels, (total_probs >= 0.5).astype(int))

    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    roc_auc = roc_auc_score(total_labels, total_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    sorted_indices = np.argsort(recall_vals)
    recall_vals = recall_vals[sorted_indices]
    precision_vals = precision_vals[sorted_indices]
    pr_auc = auc(recall_vals, precision_vals)

    logging.info(f"ROC AUC: {roc_auc:.4f}")

    # return total_probs, sample_indices, accuracy, precision, recall, f1, pr_auc, total_labels
    return total_probs, sample_indices, accuracy, precision, recall, f1, pr_auc

def accuracy(true_labels, preds):
    return np.mean(true_labels == preds)


modeling =GCNNetmuti

model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[1]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0005
LOG_INTERVAL = 160
NUM_EPOCHS =100

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets

print('\nrunning on ', model_st + '_')




log_filename = f'training_1.log'

logging.basicConfig(filename=log_filename, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

# processed_data_file_train = 'data/processed/' + '_train1'+'.pt'
# processed_data_file_test = 'data/processed/' + '_test1'+'.pt'
processed_data_file_train = 'data/processed/' + 'train1'+'.pt'
processed_data_file_test = 'data/processed/' + 'test1'+'.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    print('please run process_data_old.py to prepare data in pytorch format!')
else:
    # train_data = TestbedDataset(root='data', dataset='_train1')
    # test_data = TestbedDataset(root='data', dataset='_test1')
    train_data = TestbedDataset(root='data', dataset='train1')
    test_data = TestbedDataset(root='data', dataset='test1')
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE,shuffle=True,drop_last=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE,shuffle=False,drop_last=True)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)

    loss_fn = nn.BCELoss()  # for classification
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_acc= 0
    best_epoch = -1
    model_file_name = 'model_' + model_st + '_' + '.model'
    result_file_name = 'result_' + model_st + '_' + '.csv'

    for epoch in range(NUM_EPOCHS):
        train(model, device, train_loader, optimizer, epoch + 1)
        #  return total_probs, sample_indices, accuracy, precision, recall, f1, pr_auc, total_labels
        probs, indices, accuracy, precision, recall, f1, pr_auc = predicting(model, device, test_loader)
        print('Epoch: ', epoch, 'Accuracy: ', accuracy, 'Precision: ', precision, 'Recall: ', recall, 'F1 Score: ', f1,)

        # TODO
        # Save sorted results

        logging.info('Top 30 predictions saved to top_30_predictions.csv')



        if recall  > best_acc:
            torch.save(model.state_dict(), model_file_name)
            # Sort by probability in descending order and select top 30
            # Create a DataFrame to sort and save results

            best_epoch = epoch + 1
            best_acc = recall
            print('acc improved at epoch ', best_epoch, '; best_acc:', best_acc, model_st)
        else:
            print('No improvement since epoch ', best_epoch, '; best_acc:', best_acc, model_st)
