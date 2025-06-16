import numpy as np
from optuna.samplers import GPSampler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import optuna
import cudf
from cuml.preprocessing import train_test_split
from cuml.preprocessing import StandardScaler
import cupy as cp


def bayes(training_data):
    cols = ['time', 'longitude', 'latitude', 'xco2', 'cams', 'gosat', 'ndvi', 'evi', 'd2m', 't2m', 'skt', 'swvl1',
            'ssrd', 'u10', 'v10', 'sp', 'odiac', 'gfed', 'npp']
    df = cudf.read_csv(training_data)
    df = df.iloc[:, :]
    df.columns = cols
    df.dropna(inplace=True)
    X_tmp = df.drop(['time', 'xco2', 'longitude', 'latitude'], axis=1)
    std = StandardScaler()
    X_data = std.fit_transform(X_tmp)
    X = X_data
    y = df[['xco2']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'n_jobs': -1,
            'eval_metric': 'rmse',
            'n_estimators': trial.suggest_int('n_estimators', 6000, 15000, step=1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 1, 14),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'random_state': 42,
            'tree_method': 'hist',
            'device': 'cuda',
            'early_stopping_rounds': 50
        }

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        rmse_list = []
        r2_list = []

        for train_idx, valid_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr.to_cupy(), y_tr.to_cupy(),
                eval_set=[(X_val.to_cupy(), y_val.to_cupy())],
                verbose=False
            )

            y_pred = model.predict(X_val.to_cupy())
            y_pred_np = cp.asnumpy(y_pred)
            y_val_np = y_val.to_numpy()

            rmse_list.append(np.sqrt(mean_squared_error(y_val_np, y_pred_np)))
            r2_list.append(r2_score(y_val_np, y_pred_np))

        mean_rmse = np.mean(rmse_list)
        mean_r2 = np.mean(r2_list)

        trial.set_user_attr("rmse", mean_rmse)
        trial.set_user_attr("r2", mean_r2)
        return mean_rmse

    sampler = GPSampler(seed=42)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler
    )
    study.optimize(objective, n_trials=500, show_progress_bar=True)

    print("Finished trials:", len(study.trials))
    best = study.best_trial
    print(f"Best RMSE: {best.user_attrs['rmse']:.4f}")
    print(f"Best R2:   {best.user_attrs['r2']:.4f}")
    print("Best Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")


def main():
    training_data = ''
    bayes(training_data)


if __name__ == "__main__":
    main()
