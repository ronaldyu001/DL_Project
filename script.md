# Src Folder Walkthrough Script

Today I am going to quickly walk through the `src` folder.

The goal of this folder is to train the four base models and turn their predictions into CSV files for the ensemble model.

The four base models are the feed forward neural network, XGBoost, autoencoder, and Isolation Forest.

The folder has three main parts.

`src/general_helpers/data_pipeline.py` handles the dataset loading, train/eval/test splits, k-fold indices, and scaling.

The important part here is leakage safety. The scalers fit only on train data. Also, the train predictions for the ensemble are made with k-fold out-of-fold predictions, so each train row is predicted by a model that did not train on that row.

`src/base_models` has one folder for each model.

Each model folder has the model class file and a test or finetuning file. The model class handles the actual train, score, and eval behavior. The finetuning file handles the full loop: split the data, optionally run Optuna search, train fold models, train one final model, save metrics, save plots, and return predictions.

The shared files in `src/base_models` support all four models.

`metrics.py` calculates metrics like precision, recall, F1, ROC AUC, and average precision.

`artifacts.py` creates folders and saves CSVs, plots, and model outputs.

`tuning.py` runs Optuna search and saves the best configs and search results.

The main entry point is `src/test_base_models.py`.

This file is intentionally short. It loads `data/creditcard.csv`, runs whichever base models we select, checks that their labels line up, and stacks their prediction columns into meta learner files.

The main outputs are:

- [ ] `meta_x_train.csv`
- [ ] `meta_x_eval.csv`
- [ ] `meta_x_test.csv`
- [ ] `y_train.csv`
- [ ] `y_eval.csv`
- [ ] `y_test.csv`
- [ ] `meta_feature_names.csv`

Each column in `meta_x_train.csv` is one base model's prediction feature. So if all four models run, the meta learner gets four input columns.

To run all models, use:

```bash
python src/test_base_models.py --models all
```

To run only selected models, use something like:

```bash
python src/test_base_models.py --models ffn xgboost
```

For a faster run without hyperparameter search:

```bash
python src/test_base_models.py --models all --no-search
```

The prediction CSVs go into `src/base_models/outputs`.

Saved models go into `models`.

Metrics, plots, search results, and best configs go into `results`.

So overall, the flow is: load the data, make leakage-safe splits, train the base models, create out-of-fold train predictions, create eval and test predictions, and save everything as CSVs for the ensemble learner.
