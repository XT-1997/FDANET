from .fdanet import FDANet
#from models import resnet

def get_model(name, dataset):
    return {
            'fdanet' : FDANet(dataset=dataset)
           }[name]
