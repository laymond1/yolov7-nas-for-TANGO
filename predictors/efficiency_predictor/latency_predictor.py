'''
The currently written version uses lookup table.
It will be replaced with code of a trained predictor network later.
'''
import json
import os

class LatencyPredictor:
    def __init__(self, target):
        path = os.path.dirname(os.path.realpath(__file__))
        bfile_path = os.path.join(path, "./latency/backbone.json")
        hfile_path = os.path.join(path, "./latency/head.json")

        with open(bfile_path, "r") as f:
            self.backbone_lut = json.load(f)
        with open(hfile_path, "r") as f:
            self.head_lut = json.load(f)
    
    def predict_efficiency(self, arch):
        encoding = arch['d']
        
        b_key = self.encToKey(encoding[:4])
        h_key = self.encToKey(encoding[4:])
        latency_b = self.backbone_lut[b_key]
        latency_h = self.head_lut[h_key]

        return latency_b + latency_h
    
    def encToKey(self, encoding):
        result = ""
        for s in encoding:
            result += str(s)
        return result
