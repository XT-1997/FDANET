from .seven_scenes import SevenScenes
from .twelve_scenes import TwelveScenes
from .my_dataset import my_dataset

def get_dataset(name):
    return {
            '7S' : SevenScenes,
            '12S' : TwelveScenes,
            'my': my_dataset
           }[name]
