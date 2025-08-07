import pathlib

import torch
import torch.nn as nn
import torchvision
from torchaudio.pipelines import HUBERT_BASE
from transformers import ASTFeatureExtractor, ASTModel


class ResNetClassifier(nn.Module):
    def __init__(self, model_type, pretrained=False, num_classes=None, multi_label=False):
        super().__init__()

        if model_type.startswith('resnet50'):
            weights = torchvision.models.ResNet50_Weights.DEFAULT
            self.resnet = torchvision.models.resnet50(weights=weights if pretrained else None)
        elif model_type.startswith('resnet152'):
            weights = torchvision.models.ResNet152_Weights.DEFAULT
            self.resnet = torchvision.models.resnet152(weights=weights if pretrained else None)
        elif model_type.startswith('resnet18'):
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            self.resnet = torchvision.models.resnet18(weights=weights if pretrained else None)
        else:
            assert False

        self.linear = nn.Linear(in_features=1000, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = x.unsqueeze(1)      # (B, F, L) -> (B, 1, F, L)
        x = x.repeat(1, 3, 1, 1)    # -> (B, 3, F, L)
        x /= x.max()            # normalize to [0, 1]
        # x = self.transform(x)

        x = self.resnet(x)
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits


class VGGishClassifier(nn.Module):
    def __init__(self, sample_rate, num_classes=None, multi_label=False):
        super().__init__()

        self.vggish = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vggish.postprocess = False
        self.vggish.preprocess = False

        self.linear = nn.Linear(in_features=128, out_features=num_classes)

        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

        self.sample_rate = sample_rate

    def forward(self, x, y=None):
        batch_size = x.shape[0]
        x = x.reshape(-1, x.shape[2], x.shape[3], x.shape[4])
        out = self.vggish(x)
        out = out.reshape(batch_size, -1, out.shape[1])
        outs = out.mean(dim=1)
        logits = self.linear(outs)

        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

# TODO: understand if we fine-tune the whole model or if we freeze it and only train the last linear layer (I guess the first option)
# To note: when running the evaluation, the batch size and the learning rate have default values here that may not be adapted to the model.
# The batch size is only set to 32, which may be too small for Hubert...
class HubertClassifier(nn.Module):
    def __init__(self, num_classes=None, multi_label=False):
        super().__init__()
        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        self.hubert_base = HUBERT_BASE.get_model()
        #self.sample_rate = HUBERT_BASE.sample_rate
        
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        out, _ = self.hubert_base(x) # Will get dimension [batch size, time, num_features]
        out = torch.mean(out, dim=1) # Dimension [batch_size, num_features]
        logits = self.linear(out)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

#TODO: check that this actually does what I want it to do
class HubertClassifierFrozen(nn.Module):
    def __init__(self, num_classes=None, multi_label=False):
        super().__init__()
        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        self.hubert_base = HUBERT_BASE.get_model()
        
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x, y=None):
        with torch.no_grad():
            out, _ = self.hubert_base(x) # Will get dimension [batch size, time, num_features]
            out = torch.mean(out, dim=1) # Dimension [batch_size, num_features]
        
        logits = self.linear(out)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

# Frozen pilot study models
# TODO: I guess I should removethe dropout because I froze everything?
class SingleMultiTaskEncoder(torch.nn.Module):
    def __init__(self, hidden_layer_size: int = 768):
        super().__init__()
        self.embedding_model = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        self.hidden_layer_size = hidden_layer_size
        if torch.cuda.is_available():
            self.embedding_model = self.embedding_model.cuda()
        self.linearshared1 = torch.nn.Linear(
            768,
            self.hidden_layer_size,
        )
        self.activation1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(p=0.7)  # Will not be used anyway

    def forward(self, x):
        with torch.no_grad():
            x = self.embedding_model(**x).last_hidden_state
            # torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
            x = torch.mean(
                x, dim=1
            )  # output shape (batch size, hidden_size)
            x = self.linearshared1(x)
            x = self.activation1(x)
        return x

class SingleMultiTaskClassifier(torch.nn.Module):
    def __init__(self, model_type: str, num_classes=None, multi_label=False):
        super().__init__()
        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        self.encoder = SingleMultiTaskEncoder()
        model_path = pathlib.Path(__file__).parent.parent.resolve() / "data" / "shared_models" / (model_type.replace("pilot-", "") + ".pth")
        self.encoder.load_state_dict(torch.load(model_path))     
        
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y=None):
        x["input_values"] = x["input_values"].squeeze()
        out = self.encoder(x)
        logits = self.linear(out)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class ASTClassifierFrozen(torch.nn.Module):
    def __init__(self, num_classes=None, multi_label=False):
        super().__init__()
        self.ast = ASTModel.from_pretrained(
            "MIT/ast-finetuned-audioset-10-10-0.4593"
        )
        if torch.cuda.is_available():
            self.ast = self.ast.cuda()
        self.linear = nn.Linear(in_features=768, out_features=num_classes)
        
        if multi_label:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x["input_values"] = x["input_values"].squeeze()
        with torch.no_grad():
            x = self.ast(**x).last_hidden_state
            x = torch.mean(
                x, dim=1
            )
        logits = self.linear(x)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits
        