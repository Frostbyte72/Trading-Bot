import matplotlib.pyplot as plt
import NN_Model_2 as NN

def test_activation_functions():
    print('Testing Diffferent Activation Functions')
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
    print('Testing Diffferent batch sizes')
    x = list(range(2,128,4))
    results = []
    for i in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',batch_size = i,epochs = 150)
        results.append(result[0])

    plt.clf()
    print(x)
    print(results) 
    plt.plot(x,results,color= 'deeppink')
    plt.xlabel('Batch Size')
    plt.ylabel('Loss (MSE)')
    plt.savefig('Batch.png')

def test_epoch_size():
    print('Testing Diffferent epoch sizes')
    x = list(range(10,251,10))
    results = []
    for i in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',epochs = i)
        results.append(result[0])

    plt.clf()
    print(x)
    print(results)
    plt.plot(x,results,color= 'indigo')
    plt.xlabel('Epoch Size')
    plt.ylabel('Loss (MSE)')
    plt.savefig('Epoch.png')

#untested
def test_optimisers():
    print('Testing Diffferent optimisers')
    optimisers = ['sgd','rmsprop','adam','adamw','adadelta','adagrad','adamax','adafactor','nadam','ftrl']
    results= []

    for i in optimisers:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',optimiser = i)
        results.append(result[0])

    print(results)
    plt.clf() 
    plt.bar(optimisers,results,color= 'green')
    plt.xlabel('Gradient Decent Function')
    plt.ylabel('Loss (MSE)')
    for i in range(len(optimisers)):
        plt.text(i, results[i]//2, ("%.2f" % results[i]), ha = 'center')
    plt.savefig('optimisers.png')

#untested
def test_timestep():
    print('Testing Diffferent Timestep lengths')
    x = list(range(2,129,4))
    results = []
    for i in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',activation='tanh',time_step = i)
        results.append(result[0])
    
    plt.clf()
    print(x)
    print(results)
    plt.plot(x,results,color= 'blue')
    plt.xlabel('Timestep')
    plt.ylabel('Loss (MSE)')
    plt.savefig('Timestep.png')


def test_layer_size():
    print('testing different layer sizes')

    x = []
    #exponential increase in range
    for i in range(1,10):
        x.append(2 ** i)

    results = []
    for size in x:
        result = NN.main(plot = False ,symbol = 'GOOGL',layer_size = size)
        results.append(result[2])

    plt.clf()
    print(x)
    print(results)
    plt.plot(x,results,color= 'blue')
    plt.xlabel('Layer_size')
    plt.ylabel('Loss (MAE)')
    plt.savefig('Layer.png')

#loss function testing
#test_activation_functions()
#test_batch_size()
#test_epoch_size()
#test_timestep()
#test_optimisers()
test_layer_size()