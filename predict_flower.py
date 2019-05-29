# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import seaborn as sb
from PIL import Image
import torch
import numpy
import numpy as np
from preprocess_data import preprocess_data, cat_to_names
from get_input_args import get_input_args

train_data, valid_data, test_data, trainloader, validloader, testloader = preprocess_data()

input_args = get_input_args()
topk = input_args.top_k
# image_path = input_args.image_path 

cat_to_name = cat_to_names()    
   
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    ''' 
    """gets the image"""
    image = Image.open(image_path)

    """resizes the image in place; """
    image.thumbnail((256,256))#the bigger size is going to be 256

    """crops the image to size 224X224 """
    if image.size[0] == 256:
        w, h = 128, image.size[1]/2
    else:
        w, h = image.size[0]/2, 128
    image = image.crop((w-112, h-112, w+112, h+112))

    """changes image to NumPy matrix"""
    image = np.array(image)

    """changes colors values in matrix, integers 0-255 to float 0-1"""
    image = image / 255

    """normalize color channels"""
    means = np.array([0.485, 0.456, 0.406]) #means
    std = np.array([0.229, 0.224, 0.225]) #standard deviations

    #subtracts the means from each color channel, then divide by the standard deviation
    col_chan1 = (image[:,:,0] - means[0]) / std[0] #rather than image = (image - means) / std  
    col_chan2 = (image[:,:,1] - means[1]) / std[1]
    col_chan3 = (image[:,:,2] - means[2]) / std[2]
    image[:,:,0] = col_chan1
    image[:,:,1] = col_chan2
    image[:,:,2] = col_chan3

    image = image.transpose((2, 0, 1)) 
    return image


def predict(model, image_path):
    ''' Predict the class (or classes) of an image using a trained deep learning model and saved parameters.
    '''       
    #changes image NumPy array to tensor, adds size of a batch (one image batch) and casts input to torch.FloatTensor 
    image = torch.unsqueeze(torch.from_numpy(process_image(image_path)), 0).float()
    #allows user to use the GPU to calculate the predictions
    if torch.cuda.is_available() and input_args.gpu == 'cuda':
        model = model.to('cuda')
        image = image.to('cuda')
    else:
        model = model.to('cpu')
    
    model.eval() #in evaluation mode, dropout is turned off.
    output = model.forward(image) 
    prob = torch.exp(output)
    
    #gets the topk largest values to the tensors
    top_prob, top_prob_idx = torch.topk(prob, topk)
    #changes tensors to NumPy arrays
    top_prob = top_prob.cpu()
    top_probabilities = top_prob.data.numpy()    
    top_prob_idx = top_prob_idx.cpu()
    top_prob_idx = top_prob_idx.data.numpy()
    
    #gets classes for topk indices
    i_to_cl = {i:c for c,i in model.class_to_idx.items()} #gets dict index to class
    top_classes = [i_to_cl[i] for i in top_prob_idx.tolist()[0]]
    
    pred_prob = max(top_probabilities.tolist()[0])
    pred_name = [cat_to_name[cl] for cl in top_classes][0]

    return pred_name, pred_prob, top_probabilities, top_classes


# def imshow(image, ax=None, title=None):
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     # PyTorch tensors assume the color channel is the first dimension
#     # but matplotlib assumes is the third dimension
#     image = image.transpose((1, 2, 0))
    
#     # Undo preprocessing
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     image = std * image + mean
    
#     # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
#     image = np.clip(image, 0, 1)
    
#     ax.imshow(image)
    
#     return ax


# def show_result(probs, classes):
#     '''Prints flower_to_classify_name and picture, than plot top probabilities to names of top classes'''
#     """Gets names of top classes"""
#     top_flowers_names = [cat_to_name[e] for e in classes]

#     #gets name of testing flower
#     # is not a flower name from the user      flower_to_classify_name = cat_to_name[image_path.split("/")[-2]]

#     plt.figure(figsize= [10, 3])

#     """Makes subplot for image"""
#     axis1 = plt.subplot(1, 2, 1)

#     # is not a flower name from the user      plt.xlabel(flower_to_classify_name) 
#     imshow(process_image(), ax=axis1)          #plt.subplot(1, 2, 1)) #, title=flower_to_classify_name)

#     """makes subplot for top data"""
#     plt.subplot(1, 2, 2)
#     sb.barplot(x=probs.tolist()[0], y=top_flowers_names, color=sb.color_palette()[1]);
#     plt.xticks(rotation=90);
    
#     plt.show()
