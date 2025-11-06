# Repository Guidelines

## Project Structure & Modules
- bayes_op.py: Bayesian optimization of XGBoost via Optuna GP sampler; 10-fold CV; reports RMSE and R2; call the bayes function.

- train_and_shap.py: Trains XGBRegressor, computes SHAP, prints metrics, writes trained_model.pkl.

- data_preprocessing.py: Builds training dataset ; edit paths in main.

  For code related to interpolation, see https://github.com/SamsungSAILMontreal/ForestDiffusion

- predict.py: Runs inference on a lat/lon grid and writes NetCDF outputs; edit parameters in main.

- load/: Dataset readers and utilities (e.g., cams_load, era5_load, create_grid).

- data/ (not tracked): place input file here.



