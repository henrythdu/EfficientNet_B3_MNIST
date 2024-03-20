import torch
import torchvision

from torch import nn

def create_effnetb3_model(num_classes:int=10, 
                             seed:int=42):
    """Creates an efficientnet  feature extractor model and transforms.

    Args:
        num_classes (int, optional): number of classes in the classifier head. 
            Defaults to 42.
        seed (int, optional): random seed value. Defaults to 42.

    Returns:
        model (torch.nn.Module): EffNetB3 feature extractor model. 
        transforms (torchvision.transforms): EffNetB3 image transforms.
    """
    # Create EffNetB3 pretrained weights, transforms and model
    weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b3(weights = weights)

    # Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    # Change classifier head with random seed for reproducibility
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace = True),
    nn.Linear(in_features=1536, out_features= 10, bias = True)
)

    
    return model, transforms