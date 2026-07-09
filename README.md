# ML Fraud Detection

This project trains a simple machine learning model to detect fraudulent transactions from a synthetic dataset.

The workflow in [run.py](run.py) does the following:

1. Loads [synthetic_fraud_dataset.csv](synthetic_fraud_dataset.csv).
2. Prints a quick data audit so you can inspect the raw dataset.
3. Cleans the data and creates time-based features from the `Timestamp` column.
4. Saves the cleaned output to [cleaned_fraud_dataset.csv](cleaned_fraud_dataset.csv).
5. Trains a decision tree fraud classifier with group-aware splitting by `User_ID`.
6. Evaluates the model with accuracy, confusion matrix, classification report, and ROC AUC.
7. Runs a few sample predictions with the trained pipeline.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The project uses:

- `pandas`
- `numpy`
- `scikit-learn`

## Run

Execute the full pipeline with:

```bash
python run.py
```

Make sure the CSV files are in the project root before running the script.

## Output

Running the script will print dataset diagnostics, training metrics, and sample fraud predictions to the terminal. It will also regenerate [cleaned_fraud_dataset.csv](cleaned_fraud_dataset.csv).

## Usage Example

```bash
pip install -r requirements.txt
python run.py
```

When the script finishes, you should see:

- a data summary for the synthetic dataset,
- cleaning logs and the saved cleaned CSV notice,
- training and testing metrics for the fraud model,
- sample predictions for a few manual test transactions.

## Project Files

- [run.py](run.py): end-to-end data cleaning, model training, evaluation, and sample prediction script.
- [synthetic_fraud_dataset.csv](synthetic_fraud_dataset.csv): source dataset.
- [cleaned_fraud_dataset.csv](cleaned_fraud_dataset.csv): cleaned dataset produced by the script.
- [requirements.txt](requirements.txt): Python dependencies.
