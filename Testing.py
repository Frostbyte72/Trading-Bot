import matplotlib.pyplot as plt
import NN_Model_2 as NN

def test_activation_functions():
    activations = ['elu','gelu','hard_sigmoid','relu','selu','sigmoid','swish','tanh']
    results = []
    for act in activations:
        print(act)
        result = NN.main(plot = False ,symbol = 'GOOGL',activation=act,epochs = 150)
        results.append(result[0])
        


    print(results)
    plt.clf() 
    plt.bar(activations,results,color= 'green')
    plt.xlabel('Activation_function')
    plt.ylabel('Loss (MSE)')
    for i in range(len(activations)):
        plt.text(i, results[i]//2, ("%.2f" % results[i]), ha = 'center')
    plt.savefig('Activation.png')

def test_batch_size():
    x = list(range(2,128,4))
    results = []
    for i in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',batch_size = i,epochs = 150)
        results.append(result[0])

    plt.clf()
    print(x)
    print(results) 
    plt.plot(x,results,color= 'deeppink')
    plt.xlabel('Batch Size 2-128')
    plt.ylabel('Loss (MSE)')
    plt.savefig('Batch.png')

def test_epoch_size():
    x = list(range(10,251,10))
    results = []
    for i in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',epochs = i)
        results.append(result[0])

    plt.clf()
    print(x)
    print(results)
    plt.plot(x,results,color= 'indigo')
    plt.xlabel('Batch Size 2-128')
    plt.ylabel('Loss (MSE)')
    plt.savefig('Epoch.png')

def test_learning_rate():

    results= []

def test_timestep():
    x = list(range(2,129,4))
    results = []
    for i in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',epochs = i)
        results.append(result[0])
    
    plt.clf()
    print(x)
    print(results)
    plt.plot(x,results,color= 'blue')
    plt.xlabel('Batch Size 2-128')
    plt.ylabel('Loss (MSE)')
    plt.savefig('Epoch.png')

#test_activation_functions()
test_batch_size()
test_epoch_size()