import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_csv(file_path):
    loss_frame = pd.read_csv(file_path)
    train_loss = loss_frame['train_loss']
    x_axis = loss_frame['Unnamed: 0']
    dev_loss = loss_frame.dropna()['dev_loss']

    return x_axis, train_loss, dev_loss

if __name__=='__main__':
    paths = ['lr_10e2.csv','lr_10e3.csv','lr_10e4.csv']
    colset = ['r', 'g', 'b']
    
    plt.figure()

    for i in range(len(paths)):
        curr_x_axis, curr_train_loss, curr_dev_loss = read_csv('Project/'+paths[i])
        # plt.plot(curr_x_axis, np.log(curr_train_loss), color=colset[i], label=paths[i][:-4], alpha=0.5)
        plt.subplot(3,1,i+1)
        plt.plot(curr_dev_loss, color=colset[i], label=paths[i][:-4], alpha=0.5)

        plt.legend()
    plt.show()