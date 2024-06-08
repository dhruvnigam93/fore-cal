# Forecal

Forecal is a machine learning project that benchmarks various calibration measures on 43 datasets from the UCI Machine
Learning Repository. 

## Installation

This project uses Python and pip for package management. To set up the environment on your machine, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the necessary dependencies by running the following command:

```bash
pip install -r requirements.txt
```

Running the Project
The entry point to the project is main.py. To run the project, use the following command:

```bash
python main.py
```

### What does main.py do?

main.py performs the following tasks:

- Loads a list of datasets from a JSON file.
- Defines a set of model parameters.
- Runs a function called get_results on each dataset in a multithreaded manner.
- Writes a JSON file containing the results of the runs. The file is located in the data/results directory and is named
  results_{timestamp}.json, where {timestamp} is the current date and time.
- Prints summary metrics including the median improvements in Expected Calibration Error (ECE) and Area Under the
  Curve (AUC) across all datasets.
