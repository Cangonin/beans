

import sys
from ast import Dict, List
from pathlib import Path

import pandas as pd


def get_dict_results():
    dict_results = {}
    for model_info in MODELS:
        model = model_info[0]
        dict_results[model] = {}
        for task in TASKS:
            dataset = task[1]
            dict_results[model][dataset] = ""
    return dict_results

def get_test_results(dataset_name: str, model_type: str) -> str:
    log_file_name = dataset_name + "-" + model_type
    log_path = Path(__file__).parent.parent.resolve() / "logs" / log_file_name
    score: List[str] = []

    if not log_path.exists():
        print("Log path {log_path} does not exist, skipping", file=sys.stderr)
        return ""
    with open(log_path, 'r') as f:
        for line in f:
            if "test_metric" in line:
                score.append(line.rstrip())
    assert len(score) <= 1
    if len(score) == 0:
        print(f"No test score found for the {log_file_name}", file=sys.stderr)
        return ""
    else:
        test_score = score[0].split()[-1]
        test_score = str(round(float(test_score), 3))
        return test_score


def get_test_results_one_model(model_type: str, dict_results: Dict) -> Dict:
    for task in TASKS:
        dataset = task[1]
        dict_results[model_type][dataset] = get_test_results(dataset_name=dataset, model_type=model_type)
    return dict_results

def get_test_results_all_models(output_path: Path) -> Dict:
    dict_results = get_dict_results()
    for model_info in MODELS:
        model = model_info[0]
        dict_results = get_test_results_one_model(model_type=model, dict_results=dict_results)
    dict_results_pd = pd.DataFrame(dict_results)
    dict_results_pd = dict_results_pd.transpose()
    dict_results_pd.to_csv(output_path)

if __name__ == "__main__":
    MODELS = [
    ('hubert', 'hubert', ''),
    # ('hubert-frozen', 'hubert-frozen', ''),
    ('pilot-individual', 'pilot-individual', ''),
    ('pilot-species', 'pilot-species', ''),
    ('pilot-vox-type', 'pilot-vox-type', ''),
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
    output_path = Path(__file__).parent.parent.resolve() / "data" / "results_benchmark.csv"
    get_test_results_all_models(output_path)