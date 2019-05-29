import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from get_input_args import get_input_args
from preprocess_data import preprocess_data, cat_to_names

train_data, valid_data, test_data, trainloader, validloader, testloader = preprocess_data()
input_args = get_input_args()
lr = input_args.l_rate
model_name = input_args.arch
cat_to_name = cat_to_names()
  
#dict of pretrained networks
cnn_models = {'vgg16': models.vgg16(pretrained=True), 'densenet121': models.densenet121(pretrained=True), 
                  'resnet18': models.resnet18(pretrained=True)}

def model():
    """Gets model choosen by user"""
    model = cnn_models[model_name]
    return model


model = model()
#freeze parameters
for parameter in model.parameters():
    parameter.requires_grad = False
 
def classifier_hyperparam():
    """gets hyperparameters provided by user"""
    if model_name == 'vgg16':
        input_size = model.classifier[0].in_features
    if model_name == 'densenet121':
        input_size = model.classifier.in_features
    if model_name == 'resnet18':
        input_size = model.fc.in_features
    hidden_layers = input_args.h_layers
    output_size = len(cat_to_name)
    dropout_prob = input_args.drop_prob
    return input_size, hidden_layers, output_size, dropout_prob

input_size, hidden_layers, output_size, dropout_prob = classifier_hyperparam()


class Classifier_net(nn.Module):
    """Builds a new classifier"""
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.2):
        ''' Arguments:
            input_size: integer, size of the input
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
            drop_p: float between 0 and 1, dropout probability
        '''
        super().__init__()
        # sizes of the hidden layers; Adds the first hidden layer 
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # a variable layer_sizes; number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        
        # all hidden layers
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        #output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        #dropout regularization for reducing overfitting
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        # Forward through each layer in `hidden_layers`, with ReLU activation and dropout
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            
            x = self.dropout(x) #Since we're using dropout in the network, we need to turn it off during inference!
                        #the network will appear to perform poorly because many of the connections are turned off
        
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)  #this forward method returning the log-softmax for the output!

def model_classifier():   
    #assign classifier to the model
    model.classifier = Classifier_net(input_size, output_size, hidden_layers, drop_p=dropout_prob)
    #model.classifier = Classifier_net(1024, 102, [900, 700], drop_p=0.2)
    return model.classifier


def model_criterion():
    """Defines the loss function"""
    criterion = nn.NLLLoss()
    return criterion


def model_optimizer():
    """Define the optimizer"""
    #using the Adam optimizer (a variant of SGD) which includes momentum
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    return optimizer
