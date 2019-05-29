# Imports python modules
from time import time
# Imports functions created for this program
from get_input_args import get_input_args
from train import train, save_checkpoint, load_checkpoint
from predict_flower import predict
from preprocess_data import preprocess_data, cat_to_names

train_data, valid_data, test_data, trainloader, validloader, testloader = preprocess_data()

"""Checking image using the trained network """
def check_flower(image_path):
    start_time = time()
    input_args = get_input_args()
    cat_to_name = cat_to_names()
    topk = input_args.top_k
        
    """gets model and class_to_idx from checkpoint:"""
    model, model.class_to_idx = load_checkpoint()
                
    """gets predicted name, max probability, topk probabilities and topk classes for test image:"""
    pred_name, pred_prob, top_probabilities, top_classes = predict(model, image_path)
    
    """Printing results:"""

    print(" Pred_Name: {:20} Probability: {:0.3f}\n Top {} probabilities: {}\n Top {} classes: {}".format(pred_name, pred_prob, topk, top_probabilities, topk, top_classes))

    for i in range(1,topk):
        print(" #{}: {:20} probability: {:0.5f}".format(i+1, [cat_to_name[cl] for cl in top_classes][i], top_probabilities.tolist()[0][i]))        
    
    """Computes overall runtime and prints it in hh:mm:ss format:"""
    end_time = time()
    tot_time = end_time - start_time 
    print("\n** Total Elapsed Runtime:",
          str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
          +str(round((tot_time%3600)%60)) )
    
    return pred_name, pred_prob, top_classes[0]
