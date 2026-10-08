"""Correcao condicional do consenso Focus: previsoes out-of-fold do erro (vies, ElasticNet, XGBoost)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .configuracao import DIR_PROCESSADO, HORIZONTES_MESES
from .features_ml import preparar_matriz_features
from .splits import folds_seguros, separar_fit_val

ORDEM = ["ipca", "cambio", "igpm", "desocupacao", "selic"]
COL_DATA = {n: "referencia_periodo" for n in ORDEM} | {"selic": "data_reuniao"}
EMBARGO = max(HORIZONTES_MESES)
N_FOLDS_EXT = {n: 5 for n in ORDEM} | {"desocupacao": 3}  # amostra menor => menos folds

GRADE_L1 = [.1, .3, .5, .7, .9, .95, .99, 1.0]
PARAMS_XGB = dict(
    n_estimators=600, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.7,
    min_child_weight=5, reg_lambda=2.0, reg_alpha=0.5,
    early_stopping_rounds=30, eval_metric="rmse", random_state=42,
)
PARAMS_XGB_SEM_ES = {k: v for k, v in PARAMS_XGB.items()
                     if k not in ("early_stopping_rounds", "eval_metric")} | {"n_estimators": 150}


def pipeline_elasticnet(folds_internos):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("modelo", ElasticNetCV(l1_ratio=GRADE_L1, cv=folds_internos, max_iter=50_000, random_state=42)),
    ])


def treinar_xgb(X_tr, y_tr, datas_tr):
    sep = separar_fit_val(datas_tr.reset_index(drop=True), EMBARGO)
    if sep is None:
        return xgb.XGBRegressor(**PARAMS_XGB_SEM_ES).fit(X_tr, y_tr)
    fit, val = sep
    return xgb.XGBRegressor(**PARAMS_XGB).fit(
        X_tr.iloc[fit], y_tr[fit], eval_set=[(X_tr.iloc[val], y_tr[val])], verbose=False)


def avaliar_indicador(painel: pd.DataFrame, nome: str) -> dict:
    """Previsoes out-of-fold do erro do Focus, com folds purgados por embargo."""
    X, colunas = preparar_matriz_features(painel, incluir_horizonte=True)
    y = painel["erro_focus"].to_numpy()
    datas = painel[COL_DATA[nome]]
    hor = painel["horizonte"].to_numpy()
    oof = {k: np.full(len(y), np.nan) for k in ("vies", "elasticnet", "xgboost")}
    for tr, te in folds_seguros(datas, N_FOLDS_EXT[nome], EMBARGO):
        media_h = pd.Series(y[tr]).groupby(hor[tr]).mean()  # vies medio por horizonte, so no treino
        oof["vies"][te] = pd.Series(hor[te]).map(media_h).to_numpy()
        datas_tr = datas.iloc[tr].reset_index(drop=True)
        folds_int = folds_seguros(datas_tr, 3, EMBARGO)
        if folds_int:
            oof["elasticnet"][te] = pipeline_elasticnet(folds_int).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        oof["xgboost"][te] = treinar_xgb(X.iloc[tr], y[tr], datas.iloc[tr]).predict(X.iloc[te])
    return {"y": y, "datas": datas, "hor": hor, "oof": oof, "X": X, "colunas": colunas}


def gerar_oof() -> None:
    """Le `painel_*.parquet` e grava `oof_media_*.parquet` em `dados/processado`."""
    for n in ORDEM:
        painel = pd.read_parquet(DIR_PROCESSADO / f"painel_{n}.parquet").reset_index(drop=True)
        r = avaliar_indicador(painel, n)
        pd.DataFrame({
            "data_alvo": r["datas"].to_numpy(), "horizonte": r["hor"], "erro_focus": r["y"],
            **{f"oof_{k}": v for k, v in r["oof"].items()},
        }).to_parquet(DIR_PROCESSADO / f"oof_media_{n}.parquet", index=False)
