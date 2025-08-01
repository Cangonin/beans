import json
import pathlib
import sys

from plumbum import FG, local
from plumbum.commands.processes import ProcessExecutionError

python = local['python']
local['mkdir']['-p', 'logs']()

MODELS = [
    # ('lr', 'lr', '{"C": [0.1, 1.0, 10.0]}'),
    # ('svm', 'svm', '{"C": [0.1, 1.0, 10.0]}'),
    # ('decisiontree', 'decisiontree', '{"max_depth": [None, 5, 10, 20, 30]}'),
    # ('gbdt', 'gbdt', '{"n_estimators": [10, 50, 100, 200]}'),
    # ('xgboost', 'xgboost', '{"n_estimators": [10, 50, 100, 200]}'),
    # ('resnet18', 'resnet18', ''),
    # ('resnet18-pretrained', 'resnet18-pretrained', ''),
    # ('resnet50', 'resnet50', ''),
    # ('resnet50-pretrained', 'resnet50-pretrained', ''),
    # ('resnet152', 'resnet152', ''),
    # ('resnet152-pretrained', 'resnet152-pretrained', ''),
    # ('vggish', 'vggish', ''),
    # ('hubert', 'hubert', ''),
    # ('hubert-frozen', 'hubert-frozen', ''),
    #('pilot-individual', 'pilot-individual', ''),
    #('pilot-species', 'pilot-species', ''),
    #('pilot-vox-type', 'pilot-vox-type', ''),
    ('pilot-mtl-equal', 'pilot-mtl-equal', ''),
    ('pilot-mtl-manual', 'pilot-mtl-manual', ''),
    ('pilot-mtl-gradnorm', 'pilot-mtl-gradnorm', ''),
]

TASKS = [
    ('classification', 'watkins'),
    ('classification', 'bats'),
    ('classification', 'dogs'),
    ('classification', 'cbi'),
    ('classification', 'humbugdb'),
    ('detection', 'dcase'),
    ('detection', 'enabirds'),
    ('detection', 'hiceas'),
    ('detection', 'hainan-gibbons'),
    ('detection', 'rfcx'),
    ('classification', 'esc50'),
    ('classification', 'speech-commands'),
]

for model_name, model_type, model_params in MODELS:
    for task, dataset in TASKS:
        print(f'Running {dataset}-{model_name}', file=sys.stderr)
        log_path = f'logs/{dataset}-{model_name}'
        try:
            if model_type in ['lr', 'svm', 'decisiontree', 'gbdt', 'xgboost']:
                python[
                    'scripts/evaluate.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--params', model_params,
                    '--log-path', log_path,
                    '--num-workers', '4'] & FG                
            else:
                # Use the learning rate that the model has been trained on  
                lrs = '[1e-5, 5e-5, 1e-4]' 
                if model_type.startswith('pilot'):
                    config_path = pathlib.Path(__file__).parent.resolve() / "data" / "shared_models" / (model_type.replace("pilot-", "") + ".json")
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    lrs = str([config["learning_rate"]])
                python[
                    'scripts/evaluate.py',
                    '--task', task,
                    '--dataset', dataset,
                    '--model-type', model_type,
                    '--batch-size', '32',
                    '--epochs', '50',
                    '--lrs', lrs,
                    '--log-path', log_path,
                    '--num-workers', '1'] & FG
        except ProcessExecutionError as e:
            print(e)
