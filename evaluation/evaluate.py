import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import pearsonr



def evaluate(
        y_true,
        y_pred):

    """
    Calculate regression metrics.

    Metrics:
    MAE
    RMSE
    R
    R2
    MAPE
    NRMSE
    """


    y_true = np.array(
        y_true
    )


    y_pred = np.array(
        y_pred
    )



    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    r = pearsonr(
        y_true,
        y_pred
    )[0]


    r2 = r2_score(
        y_true,
        y_pred
    )


    mape = np.mean(
        np.abs(
            (y_true-y_pred)
            /
            (y_true+1e-8)
        )
    ) * 100



    nrmse = rmse / (
        np.max(y_true)
        -
        np.min(y_true)
        +
        1e-8
    )



    return {

        "MAE": round(mae,4),

        "RMSE": round(rmse,4),

        "R": round(r,4),

        "R2": round(r2,4),

        "MAPE": round(mape,4),

        "NRMSE": round(nrmse,4)

    }