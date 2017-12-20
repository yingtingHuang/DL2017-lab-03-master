import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets,transforms
from sklearn.externals import joblib
import model

# image transforms.
tf = transforms.Compose([  
    #transforms.Resize(32), # Resize the input Image to 32x32.
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]  
    ]  
)  

# DataLoader. Combines a dataset and a sampler, and provides single- or multi-process iterators over the dataset.  
# download and read the training data 
trainLoader =torch.utils.data.DataLoader( datasets.MNIST(root = 'D:\MINIST',
                                                         train = True, 
                                                         transform=tf,
                                                         download=False))
# download and read the test data 
testLoader = torch.utils.data.DataLoader(datasets.MNIST(root = 'D:\MINIST',
                                                        train = False,
                                                        transform=tf,
                                                        download=False))
trainNum = int(len(trainLoader))
testNum =  int(len(testLoader))
print('Number of training samples: ', trainNum )
print('Number of test samples: ', testNum )

net = model.LeNet5(1,10) # Load LeNet5 network 
criterion = nn.CrossEntropyLoss() # CrossEntropyLoss
optimizer = optim.SGD(net.parameters(), 
                      lr=0.001, 
                      momentum=0.9)  # SGD 
                      
# train
useCuda= True
iterations=5
for epoch in range(iterations): # iteration 
    correct = 0.0
    trainLoss = 0.0
    for idx, (inputs, target) in enumerate(trainLoader):
        if useCuda :
            inputs,target=inputs.cuda(),target.cuda()
        inputs,target = Variable(inputs),Variable(target) # wrap them in Variable        
        optimizer.zero_grad() # initialization 
        outputs = net.forward(inputs) # forward propagation 
        loss = criterion(outputs,target) # calculate the loss
        loss.backward() # backward propagation 
        optimizer.step() # complete the gradient descent optimization process 
        pred = outputs.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum() 
        trainLoss += loss
    print('Train Epoch: {}\tAve Loss: {:.6f}\taccuracy: {:.4f}%'.format(
           epoch, trainLoss.data[0]/trainNum, 100.* correct/trainNum ))     

# test
validationLoss = 0.0
correct = 0.0
for idx, (inputs, target) in enumerate(testLoader): 
    inputs,target = Variable(inputs),Variable(target) # wrap them in Variable   
    outputs = net.forward(inputs) # forward
    validationLoss += criterion(outputs,target) # calculate the loss
    pred = outputs.data.max(1)[1] # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum() 

validationLoss /= testNum # Average loss
correct /=testNum # Accuracy
print('validationSet: \tAve loss={:.6f} \tAccuracy={:.4f}%'.format(validationLoss.data[0], 100.*correct) )

# Save the best model as a .pkl file 
joblib.dump(net, 'LeNet5.pkl') 