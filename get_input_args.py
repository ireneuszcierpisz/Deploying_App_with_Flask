import argparse

def get_input_args():
    """Gets inputs from user"""
    parser = argparse.ArgumentParser(description='Process inputs for image classifier.')

    parser.add_argument('--saved_model_path' , type=str, default='d:/Users/Irek_git/irek-image-classifier-app/checkpoint/flower102_checkpoint.pth', help='path of saved model')
    parser.add_argument('--dir', type=str, default='d:/Users/Irek_git/irek-image-classifier-app/flowers/', help='path to the folder of flowers')
    parser.add_argument('--arch', type=str, default='densenet121', help='CNN model architecture')
    parser.add_argument('--l_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--drop_prob', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=22, help='Number of epochs')
    parser.add_argument('--h_layers', type=list, default=[900, 700], help='List of hidden units')
    parser.add_argument('--gpu_cpu', type=bool, default=False, help='Whether to use GPU. If yes type: True') 
    parser.add_argument('--gpu', type=str, default='cuda', help='GPU. The default is "cuda"')
    parser.add_argument('--image_path', type=str, default='d:/Users/Irek_git/irek-image-classifier-app/flowers/test/14/image_06083.jpg', help='Path of image to be predicted')     
    parser.add_argument('--top_k', type=int, default=3, help='Display top k probabilities and classes') 
    parser.add_argument('--category_names', type=str, default='d:/Users/Irek_git/irek-image-classifier-app/cat_to_name.json', help='path to category to flower name mapping json')    
    return parser.parse_args()
