import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error,mean_absolute_percentage_error
import xgboost as xgb

def training_model(training_data):
    cols = ['time', 'longitude', 'latitude', 'xco2', 'cams', 'gosat', 'ndvi', 'evi', 'd2m','t2m','skt','swvl1','ssrd', 'u10',
            'v10', 'sp',  'odiac', 'gfed', 'npp']
    df = pd.read_csv(training_data)
    df=df.iloc[:,:]
    df.columns = cols
    df.dropna(inplace=True)
    X_tmp = df.drop(['time','xco2','longitude', 'latitude'], axis=1)
    std = StandardScaler()
    X_data = std.fit_transform(X_tmp)
    X = pd.DataFrame(X_data, columns = X_tmp.columns)
    y = df[['xco2']]
    print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True,random_state=42)
    XGBR = xgb.XGBRegressor(n_jobs=-1, n_estimators=8000, max_depth=14, min_child_weight=7,
                            learning_rate=0.018587242305603307,
                            subsample=0.7493950181119737, colsample_bytree=0.93, gamma=0.00018886343368031545,
                            reg_alpha=0.19905108862848447, reg_lambda=0.4817462851284352,tree_method='hist',device='cuda')
    XGBR.fit(X_train, y_train)
    y_pred = XGBR.predict(X_test)
    shap_values = shap.explainers.GPUTree(XGBR).shap_values(X_test)
    print(shap.summary_plot(shap_values, X_test))
    print(shap.summary_plot(shap_values, X_test, plot_type="bar"))
    print(f'RMSE : {np.sqrt(mean_squared_error(y_test, y_pred))}')
    print(f'R2 : {sklearn.metrics.r2_score(y_test, y_pred)}')
    print(f'MAPE : {mean_absolute_percentage_error(y_test, y_pred)}')
    print('The machine learning model is successfully trained.')
    print('Mean Squared Error (MSE): ', sklearn.metrics.mean_squared_error(y_test, y_pred))
    joblib.dump(XGBR, 'trained_model.pkl')

def main():
    training_data = ''
    training_model(training_data)

if __name__ == "__main__":
    main()