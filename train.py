import dataPreprocess
import model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time


def train(model, device, train_loader, valid_loader, epochs, lf, optimizer):
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, 0)
    best_valid_acc = 0
    best_model_report = ''

    output_file = open('output.txt', 'w+')
    output_file.write('start...')
    output_file.write('\n')
    output_file.close()
    print('start...')
    # training procedure
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        loss_val = 0
        true_running = 0
        total_running = 0
        for i, data in enumerate(train_loader):
            x, gt = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            predicted = model(x)
            loss = lf(predicted, gt)

            result, predicted_class = torch.max(predicted, 1)
            true_running += (predicted_class == gt).sum()
            total_running += predicted_class.shape[0]

            loss.backward()
            optimizer.step()

            loss_val += loss.item()

        train_loss = loss_val / len(train_loader)
        accuracy = true_running / total_running
        print(f'Epoch - {epoch} Train - Loss : {train_loss} Accuracy : {accuracy}')
        output_file = open('output.txt', 'a')
        output_file.write(f'Epoch {epoch}/{epochs} - Train')
        output_file.write(f'loss: {train_loss}')
        output_file.write('\n')
        output_file.write(f'accuracy: {accuracy}')
        output_file.write('\n')
        output_file.close()

        sched.step()
        model.eval()

        # validating procedure
        valid_loss_val = 0
        valid_true_running = 0
        valid_total_running = 0
        y_pred = np.array([])
        y_test = np.array([])
        for i, data in enumerate(valid_loader):
            x, gt = data[0].to(device, dtype=torch.float32), data[1].to(device, dtype=torch.long)
            predicted = model(x)
            loss = lf(predicted, gt)

            result, predicted_class = torch.max(predicted, 1)
            valid_true_running += (predicted_class == gt).sum()
            valid_total_running += predicted_class.shape[0]

            valid_loss_val += loss.item()

            y_pred = np.append(y_pred, predicted_class.cpu().detach().numpy())
            y_test = np.append(y_test, gt.cpu().detach().numpy())

        # calculating measurements
        valid_loss = valid_loss_val / len(train_loader)
        accuracy = valid_true_running / valid_total_running
        print(f'Epoch - {epoch} Validation - Loss : {valid_loss} Accuracy : {accuracy}')

        # accuracy and loss
        output_file = open('output.txt', 'a')
        output_file.write(f'Epoch {epoch}/{epochs} - Validation')
        output_file.write(f'loss: {valid_loss / len(train_loader)}')
        output_file.write('\n')
        output_file.write(f'accuracy: {accuracy}')
        output_file.write('\n')

        # precision, recall, f1-score
        output_file.write('\nClassification Report\n')
        output_file.write(classification_report(y_test, y_pred))
        output_file.write('\n')

        # confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        output_file.write(str(conf_matrix))
        output_file.write('\n')

        # time usage for each epoch
        end_time = time.time()
        usage_time = end_time - start_time
        output_file.write(f'Time usage: {usage_time} secs')
        output_file.write('\n')
        output_file.write('\n')

        output_file.close()

        # save best model and its performance report, can be used for futher training
        if accuracy > best_valid_acc:
            best_valid_acc = accuracy
            best_model_report = classification_report(y_test, y_pred)
            torch.save(model.state_dict(), './resnet_attention.pth')

        # report the best training model
        if epoch == epochs:
            output_file = open('output.txt', 'a')
            output_file.write(f'End Training Overall Report')
            output_file.write('\n')
            output_file.write(f'Best Validation Accuracy: {best_valid_acc}')
            output_file.write('\n')
            output_file.write(f'Classification Report: {best_model_report}')
            output_file.write('\n')
            output_file.write(f'The best model is saved under resnet_attention.pth')
            output_file.close()


def main(t_csv, v_csv, b_size, t_device, lr, eps_num, ls_function, d_set):
    label_list = {"aegypti": 0, "albopictus": 1, "arabiensis": 2, "gambiae": 3, "quinquefasciatus": 4, "pipiens": 5}

    train_data = dataPreprocess.ListDataset(t_csv, label_list, "train", d_set)
    vali_data = dataPreprocess.ListDataset(v_csv, label_list, "validation", d_set)

    train_loader = DataLoader(train_data, b_size, shuffle=True)
    vali_loader = DataLoader(vali_data, b_size, shuffle=True)

    resnet_model = model.resnet18_attention(1, 6)  # 1 channel, 6 classes
    resnet_model = resnet_model.to(t_device)
    optimizer = optim.Adam(resnet_model.parameters(), lr=lr)

    train(resnet_model, t_device, train_loader, vali_loader, eps_num, ls_function, optimizer)


if __name__ == "__main__":
    train_csv = "./trainData_Abuzz.csv"
    valid_csv = "./valiData_Abuzz.csv"
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    learning_rate = 2e-4
    epochs = 2
    loss_function = nn.CrossEntropyLoss()

    # dataset name = 'Abuzz' or 'Wingbeats'
    dataset_name = 'Abuzz'

    main(train_csv, valid_csv, batch_size, device, learning_rate, epochs, loss_function)
