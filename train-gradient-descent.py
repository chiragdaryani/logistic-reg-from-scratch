# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt



def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels





def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    
    #Not Working:
    #e_x = np.exp(x - np.max(x))
    #return e_x / e_x.sum(axis=0) 
    
    #Working correctly:
    s = np.max(x, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(x - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div
    

def get_one_hot(t):
   
    t = t.reshape(t.shape[0],)

    shape = (t.size, t.max()+1)

    one_hot = np.zeros(shape)

    rows = np.arange(t.size)
    #print(rows)

    one_hot[rows, t] = 1
    #print(one_hot)
    return one_hot
    




def get_accuracy(t, t_hat):
    """
    Calculate accuracy
    """

    #print(t_hat)

    #print(t)
    #print(t.reshape(t.shape[0],))
    
    acc = np.sum(np.equal(t.reshape(t.shape[0],), t_hat)) / len(t) #shape should be same before we do comparision
    
    return acc



def cross_entropy_loss(X, w, t):

    # Get prediction
    y_prob = softmax(X@w)
    #t_pred = np.argmax(y_prob, axis=1)
    #print(t_pred)

  
    #we need one hot encoding of class label in cross entropy formula
    t_one_hot = get_one_hot(t)

    #no of records in dataset
    m =  X.shape[0]

    loss=(-1/m) * np.sum(t_one_hot*np.log(y_prob))



    #Calculate gradient of cross entropy: X.T(y-t) 

    #gradient is 1/M * X.T(y_pred-t)
    grad = (1/m) * X.T @ (y_prob-t_one_hot)


    return loss, grad



def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K


    m = X.shape[0]

    # Get prediction
    y_prob = softmax(X@W)
    t_hat = np.argmax(y_prob, axis=1)
    #print(t_pred)

    #Calculate Loss
    loss,_ = cross_entropy_loss(X, W, t)    

    #Calculate accuracy
    acc = get_accuracy(t, t_hat)

    return y_prob, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]



   
    # initialization
    w = np.zeros([X_train.shape[1], N_class]) #10 classes 
    # w: (d+1)x10

    losses_train = []
    accuracies_val = []

    w_best = None
    acc_best = 0
    epoch_best = 0

    for epoch in range(MaxEpoch):

        loss_this_epoch = 0

        for b in range(int(np.ceil(N_train/batch_size))):


            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            loss_batch, gradient = cross_entropy_loss(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            #print(loss_batch)
            
            
            # Mini-batch gradient descent
            
            #Update value of weights using learning rate and gradient

            #update weight
            w = w - alpha*gradient

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        avg_loss_for_epoch =  loss_this_epoch / (N_train/batch_size) #divide by no of batches
        #print(avg_loss_for_epoch)
        # append to list of training losses over all epochs
        losses_train.append(avg_loss_for_epoch)
        
        # 2. Perform validation on the validation set by the risk
        y_hat_val, t_hat_val, loss_val, acc_for_val = predict(X_val, w, t_val)

        #print(acc_for_val)
        accuracies_val.append(acc_for_val)  

        # 3. Keep track of the best validation epoch, risk, and the weights

        # We have to find whether this epoch score has improved the performance or not
        
        if acc_best <= acc_for_val:
            epoch_best = epoch
            acc_best = acc_for_val
            w_best = w

        
    # Best epoch, acc and w after all epochs completed
    #print(epoch_best)
    #print(acc_best)
    #print(w_best)

    # Return some variables as needed

    return epoch_best, acc_best, w_best, losses_train, accuracies_val


##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)



N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0          # weight decay


# report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)














# Report numbers and draw plots as required.

print("\nThe number of epoch that yields the best validation performance: ", epoch_best) 
print("The validation performance (accuracy) in that epoch: ", acc_best )
print("The test performance (accuracy) in that epoch: ",acc_test) 

# Visualize loss history
epochs = list(range(MaxEpoch))

plt.figure()

plt.plot(epochs, train_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.savefig('Q2.a) Training Loss.png')

plt.figure()

plt.plot(epochs, valid_accs, 'r--')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.savefig('Q2.a) Validation Accuracy.png')





#References:

#https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
#https://androidkt.com/implement-softmax-and-cross-entropy-in-python-and-pytorch/
#https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python/39558290#39558290