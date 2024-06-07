import json
from src.domain.domain_services import get_results, convert_results_to_df, get_summary_metrics

from src.domain.domain_services import white_list_uci_ids
import multiprocessing
from tqdm import tqdm
from itertools import product
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

dataset_meta_ls = [
    x
    for x in json.load(open("uci_datasets_meta1.json", "rb"))
    if x["uci_id"] in white_list_uci_ids
]

model_params = [{
    "base_nn_hidden_size": 100,
    "base_nn_hidden_layers": 2,
    "base_nn_n_epochs": 100,
    "calibration_forecal_n_bins": 50,
    "calibration_forecal_n_samples": 100,
    "calibration_forecal_criterion": "friedman_mse",
    "calibration_forecal_weight": True,
    "calibration_forecal_n_estimators": 100,
    "calibration_histbin_bins": 20,
}]

arg_list = list(product(dataset_meta_ls, model_params))
total_ = len(arg_list)
print(f"Total number of runs: {total_}")


def main():
    with multiprocessing.pool.ThreadPool(total_) as pool:
        with tqdm(total=total_) as pbar:
            result_list = []
            for result in pool.imap_unordered(get_results, arg_list):
                result_list.append(result)
                pbar.update()

    # write to file
    with open(f"data/results/results_{timestamp}.json", "w") as f:
        json.dump({"model_params": model_params[0], "data": result_list}, f)
    print(f"Written to results/results_{timestamp}.json")

    results_df = convert_results_to_df([{"model_params": model_params[0], "data": result_list}])
    metrics = get_summary_metrics(results_df)
    print("Summary metrics:")
    print(metrics)


if __name__ == "__main__":
    main()
