from sklearn import ensemble
import xgboost as xgb 
from sklearn import linear_model

MODELS = {
    "randomforest_classifier": ensemble.RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        verbose=2),

    "randomforest_regressor": ensemble.RandomForestRegressor(
        n_estimators=200,
        n_jobs=-1,
        verbose=2),

    "xgb_classifier": xgb.XGBRFClassifier(
        learning_rate=1,
        subsample=0.9,
        
    ),

    "xgb_regressor": xgb.XGBRFRegressor(
        learning_rate=1,
        subsample=0.9

    ),

    "logistic_regressor": linear_model.LogisticRegression(
        penalty='elasticnet',
        fit_intercept=True,
        class_weight='balanced',
        random_state=42,
        solver='saga',
        verbose= 2,
        n_jobs=-1,

        )


#TODO: add more models here


}