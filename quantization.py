import torch
import torchvision

# Helper function to load model from FaceDetector
def get_instance_objectdetection_model(num_classes=2,path_weight=None):
    # load an instance segmentation model pre-trained on COCO
    create_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,pretrained_backbone=True)

    # get the number of input features for the classifier
    in_features = create_model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    create_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    if path_weight is not None:
        create_model.load_state_dict(torch.load(path_weight,map_location=torch.device('cpu')))
    

    return create_model


#Loads classifier
def load_checkpoint(filepath):
  checkpoint = torch.load(filepath)
  model = checkpoint['model']
  model.load_state_dict(checkpoint['state_dict'])

  
  for parameter in model.parameters():
    parameter.requires_grad = False
    
  return model.eval()


filepath = 'classifier.pth'
loaded_model = load_checkpoint(filepath)


path_trained_weight = "customtrained_fasterrcnn_resnet50_fpn.pth"
model = get_instance_objectdetection_model(num_classes=2,path_weight=path_trained_weight)

#Quantizing detector
torch.quantization.prepare(model, inplace=True)
torch.quantization.convert(model, inplace=True)

#Quantizing classifier
torch.quantization.prepare(loaded_model, inplace=True)
torch.quantization.convert(loaded_model, inplace=True)

#Saving quantized models
torch.save(loaded_model,'quant_classifier.pth')
torch.save(model,'quant_detector.pth')
