from ast import Str
from retinaface import RetinaFace 
import torch
from  config import cfg_re50
from config import cfg_mnet


script_model = torch.jit.script(RetinaFace(cfg = cfg_mnet))

print(script_model)


