efficientnet_params = {"efficientnet-b0": (1.0, 1.0, 0.2, (224, 224, 3)),
                       "efficientnet-b1": (1.0, 1.1, 0.2, (240, 240, 3)),
                       "efficientnet-b2": (1.1, 1.2, 0.3, (260, 260, 3)),
                       "efficientnet-b3": (1.2, 1.4, 0.3, (300, 300, 3)),
                       "efficientnet-b4": (1.4, 1.8, 0.4, (380, 380, 3)),
                       "efficientnet-b5": (1.6, 2.2, 0.4, (456, 456, 3)),
                       "efficientnet-b6": (1.8, 2.6, 0.5, (528, 528, 3)),
                       "efficientnet-b7": (2.0, 3.1, 0.5, (600, 600, 3))}

efficientdet_params = {
    "efficientdet-d0" : ((512, 512, 3),    "efficientnet-b0", 3, 64,  3),
    "efficientdet-d1" : ((640, 640, 3),    "efficientnet-b1", 4, 88,  3),
    "efficientdet-d2" : ((768, 768, 3),    "efficientnet-b2", 5, 112, 3),
    "efficientdet-d3" : ((896, 896, 3),    "efficientnet-b3", 6, 160, 4),
    "efficientdet-d4" : ((1024, 1024, 3),  "efficientnet-b4", 7, 224, 4),
    "efficientdet-d5" : ((1280, 1280, 3),  "efficientnet-b5", 7, 288, 4),
    "efficientdet-d6" : ((1280, 1280, 3),  "efficientnet-b6", 8, 384, 5),
    "efficientdet-d7" : ((1536, 1536, 3),  "efficientnet-b6", 8, 384, 5),
    "efficientdet-d7x": ((1536, 1536, 3),  "efficientnet-b7", 8, 384, 5)}
