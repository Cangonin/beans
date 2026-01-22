from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


def prepare_results_dataframe(results_path: Path) -> pd.DataFrame:
    results_df = pd.read_csv(results_path, index_col=0)
    results_df = results_df.T
    results_df = results_df.rename(columns=get_mapping_model_names())
    return results_df

def plot_benchmark_results(results_path: Path):
    results_df = prepare_results_dataframe(results_path)
    colours = ["r", "r", "r", "g", "g", "g", "b"]
    models = results_df.columns
    dashes = [(1, 5), (5, 5), (3, 5, 1, 5), (1, 5), (5, 5), (3, 5, 1, 5), ()]
    for i, model in enumerate(models):
        sns.lineplot(x=results_df.index, y=results_df[model], color=colours[i], dashes=dashes[i])
    plt.xticks(rotation=30)
    plt.savefig("beans_datasets.png")
    plt.close()

def get_mapping_model_names() -> Dict[str, str]:
    mapping = {"pilot-individual": "Individual",
               "pilot-species": "Species",
               "pilot-vox-type": "Vox type",
               "pilot-mtl-equal": "Equal weights",
               "pilot-mtl-manual": "Static weights",
               "pilot-mtl-gradnorm": "GradNorm",
               "ast-frozen": "AST Frozen"}
    return mapping

def calculate_significant_differences_matrix(results_path: Path):
    results_df = prepare_results_dataframe(results_path=results_path)
    models = results_df.columns
    matrix_test_results = np.ones((len(models), len(models)))
    for i, model_A in enumerate(models):
        for j, model_B in enumerate(models):
            if model_A != model_B:
                res = wilcoxon(results_df[model_A], results_df[model_B], zero_method="pratt")
                matrix_test_results[i, j] = res.pvalue
    test_results_df = pd.DataFrame(matrix_test_results, columns=models, index=models)
    sns.heatmap(test_results_df, annot=True)
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig("matrix_significance_results.png")
    plt.close()



if __name__ == "__main__":
    results_csv = "/home/cangonin/github/beans/data/results_benchmark_new.csv"
    calculate_significant_differences_matrix(results_path=results_csv)
    # plot_benchmark_results(results_csv)