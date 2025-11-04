# config.py
class Config:
    def __init__(self):
        self.backbone = 'resnet50'
        self.pretrained = True
        self.use_sync_bn = False
        self.num_classes = 1
        self.input_size = (512, 512)
