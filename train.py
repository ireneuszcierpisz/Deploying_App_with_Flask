import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from preprocess_data import preprocess_data
from get_input_args import get_input_args
import my_model
from my_model import model, model_classifier, model_criterion, model_optimizer, classifier_hyperparam

train_data, valid_data, test_data, trainloader, validloader, testloader = preprocess_data()
input_size, hidden_layers, output_size, dropout_prob = classifier_hyperparam()
input_args = get_input_args()

model.classifier = model_classifier()
optimizer = model_optimizer()
criterion = model_criterion()

#assigne a device
if input_args.gpu_cpu:
    device = input_args.gpu
    if torch.cuda.is_available() and device == 'cuda':
        model = model.to(device)
    else:
        model = model.to('cpu')
else:
    device = 'cpu'
    model = model.to(device)
    
epochs = input_args.epochs
print_every = 40


#train the model
def train():
    """learning nn on the training data"""
    print("Computes using: ", device)
    steps = 0
    for e in range(epochs):
        model.train() #In training mode, dropout is turned on; 
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)        
            optimizer.zero_grad() #sets gradient to zero at the start of new batch 
                                  #because the backward() function accumulates gradients        
            outputs = model.forward(inputs)
            """calculate the actual loss"""
            loss = criterion(outputs, labels)#compute a gradient according to a given loss function
            loss.backward() #back propagation
            optimizer.step() #updating weeights
        
            #statistics
            running_loss += loss.item()

            if steps % print_every == 0:
                with torch.no_grad():
                    """Checks if training cost is dropping but valid accuracy is static -> model is overfitting
                    if cost of training data improves but cost of valid data is going to be worse ->model is overfitting"""
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Trainning Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation()[0]/len(validloader)),
                          "Validation Accuracy: {:.3f}.. ".format(validation()[1]/len(validloader)))

                running_loss = 0            
                # training is back on
                model.train()

    check_accuracy_on_test()  


    
def check_accuracy_on_test():  
    """Checks model accuracy on test data in mode .eval() and .no_grad() using device provided by user"""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) #gets tensor of predicted classes for max probability values from outputs
            total += labels.size(0)
            correct += (predicted == labels).sum().item()            
    print('Accuracy of the network on test data: %d %%' % (100 * correct / total))
    

def validation():
    """Checks model loss and accuracy on valid data in mode .eval() using device provided by user"""
    valid_loss = 0
    valid_accuracy = 0
    model.eval()
    for inputs, labels in iter(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        #Since the log-softmax is a log probability distibution over the classifier_net class
        # we need to take the exponential (torch.exp) of the output
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        valid_accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, valid_accuracy


"""Saves tuples mapping of flower classes values to indices of the train_data's images(flowers) predicted by model:"""
model.class_to_idx = train_data.class_to_idx


def save_checkpoint():
    """Storing information about the model architecture and parameters in dict checkpoint"""
    model.cpu()
    checkpoint = {
                  'model_arch': model,
                  'input_size': input_size,
                  'output_size': output_size,
                  'hidden_layers': hidden_layers,
                  'dropout_prob': dropout_prob,
                  'state_dict': model.state_dict(), 
                  'class_to_idx': model.class_to_idx,
                  'optimizer': optimizer.state_dict()
                 }
    torch.save(checkpoint, input_args.saved_model_path)



def load_checkpoint():
    """Loads model architecture and parameters from the checkpoint and returns model and model.class_to_idx"""
    #checkpoint = torch.load(input_args.saved_model_path)
    checkpoint = torch.load(input_args.saved_model_path, map_location='cpu')
    model = checkpoint['model_arch']
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.classifier = my_model.Classifier_net(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], checkpoint['dropout_prob'])  #checkpoint['classifier']    
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']    
    optimizer = checkpoint['optimizer']
        
    return model, model.class_to_idx