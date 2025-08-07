from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon


def prepare_results_dataframe(results_path: Path) -> pd.DataFrame:
    results_df = pd.read_csv(results_path, index_col=False)
    results_df = results_df.drop("rfcx", axis=1) # No results for this dataset
    results_df.reset_index(drop=True, inplace=True)
    results_df = results_df.set_index('model').T
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

def calculate_significant_differences_matrix(results_path: Path):
    results_df = prepare_results_dataframe(results_path=results_path)
    models = results_df.columns
    matrix_test_results = np.ones((len(models), len(models)))
    for i, model_A in enumerate(models):
        for j, model_B in enumerate(models):
            if model_A != model_B:
                res = wilcoxon(results_df[model_A], results_df[model_B])
                matrix_test_results[i, j] = res.pvalue
    test_results_df = pd.DataFrame(matrix_test_results, columns=models, index=models)
    sns.heatmap(test_results_df, annot=True)
    plt.tight_layout()
    plt.savefig("matrix_significance_results.png")
    plt.close()



if __name__ == "__main__":
    results_csv = "/home/cangonin/Documents/github/beans/data/results_benchmark.csv"
    calculate_significant_differences_matrix(results_path=results_csv)
    plot_benchmark_results(results_csv)