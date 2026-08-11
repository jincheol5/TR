import os
import torch
import torch.nn as nn

class ModelUtils:
    base_path=os.path.join("..","data","TR_Embedding","checkpoints")
    @staticmethod
    def save_model_parameter(
            model:nn.Module,
            model_name:str
        ):
        file_name=model_name+".pt"
        file_path=os.path.join(ModelUtils.base_path,file_name)
        torch.save(model.state_dict(),file_path)
        print(f"Save {model_name} model parameter!")

    @staticmethod
    def load_model_parameter(
            model:nn.Module,
            model_name:str
        ):
        file_name=model_name+".pt"
        file_path=os.path.join(ModelUtils.base_path,file_name)
        model.load_state_dict(torch.load(file_path))
        return model