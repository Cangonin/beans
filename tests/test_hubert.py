import numpy as np
import torch
import torch.optim as optim

from beans.models import HubertClassifier, HubertClassifierFrozen


# TODO: fix the test
# Maybe test with the training=True or false attribute instead of checking the weights?
def test_hubert_shallow_versus_deep_finetuning():
    num_classes = 5
    label = torch.zeros(1, dtype=torch.long)
    label[0]=2
    classifier_deep = HubertClassifier(num_classes=num_classes)
    classifier_shallow = HubertClassifierFrozen(num_classes=num_classes)
    optimizer_deep = optim.Adam(params=classifier_deep.parameters(), lr=0.001)
    optimizer_shallow = optim.Adam(params=classifier_shallow.parameters(), lr=0.001)
    
    classifier_deep.train()
    classifier_shallow.train()

    original_weights_first_layer_hubert_deep = classifier_deep.hubert_base.encoder.transformer.layers[0].attention.out_proj.weight
    original_weights_first_layer_hubert_deep = classifier_deep.hubert_base.feature_extractor.conv_layers[0].conv.weight

    original_weights_first_layer_hubert_shallow = classifier_shallow.hubert_base.encoder.transformer.layers[0].attention.out_proj.weight

    x = torch.from_numpy(np.ones((1, 10000), dtype=np.float32)) # dim [batch, time] 2 seconds of audio sampled at 16 kHz
    loss_deep, _ = classifier_deep(x, label)
    loss_shallow, _ = classifier_shallow(x, label)

    loss_deep.backward()
    loss_shallow.backward()

    optimizer_deep.step()
    optimizer_shallow.step()

    weights_first_layer_hubert_deep = classifier_deep.hubert_base.encoder.transformer.layers[0].attention.out_proj.weight
    weights_first_layer_hubert_deep = classifier_deep.hubert_base.feature_extractor.conv_layers[0].conv.weight

    weights_first_layer_hubert_shallow = classifier_shallow.hubert_base.encoder.transformer.layers[0].attention.out_proj.weight

    assert not torch.equal(original_weights_first_layer_hubert_deep, weights_first_layer_hubert_deep)
    assert torch.equal(original_weights_first_layer_hubert_shallow, weights_first_layer_hubert_shallow)