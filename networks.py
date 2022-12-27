from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision import models
from init import initialize_weight



def pretrained_network(model_name, n_classes, weight_initialization=None, **kwargs):
    model_name = model_name.lower()
    if 'vgg' in model_name:
        if model_name == 'vgg11':
            model = models.vgg11(weights='VGG11_Weights.DEFAULT')
        elif model_name == 'vgg13':
            model = models.vgg13(weights='VGG13_Weights.DEFAULT')
        elif model_name == 'vgg16':
            model = models.vgg16(weights='VGG16_Weights.DEFAULT')
        elif model_name == 'vgg19':
            model = models.vgg19(weights='VGG19_Weights.DEFAULT')
        elif model_name == 'vgg11_bn':
            model = models.vgg11_bn(weights='VGG11_BN_Weights.DEFAULT')
        elif model_name == 'vgg13_bn':
            model = models.vgg13_bn(weights='VGG13_BN_Weights.DEFAULT')
        elif model_name == 'vgg16_bn':
            model = models.vgg16_bn(weights='VGG16_BN_Weights.DEFAULT')
        elif model_name == 'vgg19_bn':
            model = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT')
    elif 'resnet' in model_name:
        if model_name == 'resnet18':
            model = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        elif model_name == 'resnet34':
            model = models.resnet34(weights='ResNet34_Weights.DEFAULT')
        elif model_name == 'resnet50':
            model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        elif model_name == 'resnet101':
            model = models.resnet101(weights='ResNet101_Weights.DEFAULT')
        elif model_name == 'resnet152':
            model = models.resnet152(weights='ResNet152_Weights.DEFAULT')
        elif model_name == 'resnext50_32x4d':
            model = models.resnext50_32x4d(weights='ResNet50_32x4d_Weights.DEFAULT')
        elif model_name == 'resnext101_32x8d':
            model = models.resnext101_32x8d(weights='ResNet101_32x8d_Weights.DEFAULT')
        elif model_name == 'resnext101_64x4d':
            model = models.resnext101_64x4d(weights='ResNet101_64x4d_Weights.DEFAULT')
        elif model_name == 'wide_resnet50_2':
            model = models.wide_resnet50_2(weights='Wide_ResNet50_2_Weights.DEFAULT')
        elif model_name == 'wide_resnet101_2':
            model = models.wide_resnet101_2(weights='Wide_ResNet101_2_Weights.DEFAULT')
    elif 'mobilenet' in model_name:
        if model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

    
    # load the pretrained model from pytorch
    # initialize weight
    if weight_initialization:
        initialize_weight(model, weight_initialization)

    if 'vgg' in model_name:
        # freeze training for all layers
        for param in model.features.parameters():
            param.require_grad = False
        # newly created modules have require_grad=True by default
        num_features = model.classifier[6].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, n_classes)]) # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features) # Replace the model classifier
    elif 'resnet' in model_name:
        # freeze training for all layers
        for param in model.parameters():
            param.require_grad = False
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, n_classes)
    elif 'mobilenet' in model_name:
        # freeze training for all layers
        for param in model.features.parameters():
            param.require_grad = False
        # newly created modules have require_grad=True by default
        num_features = model.classifier[1].in_features
        features = list(model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, n_classes)]) # Add our layer with 4 outputs
        model.classifier = nn.Sequential(*features) # Replace the model classifier

    return model
