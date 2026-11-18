#!/usr/bin/env python3
"""Age prediction: integrated XGBoost (gait) + MLP (anthropometrics), split conformal intervals.

Examples:
 python age_conformal_xgb_mlp.py --demo
 python age_conformal_xgb_mlp.py --anthro-csv participants_measures.csv --gait-csv gait_parameters_estimation.csv --id-column ID --target Age
 python age_conformal_xgb_mlp.py --csv merged.csv --id-column ID --target Age --anthro-cols Height BMI WaistC --gait-cols Speed_UGS Stride_UGS Cadence_UGS

Install: pip install numpy pandas scikit-learn xgboost
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ANTHRO = ['Sex', 'PA_level', 'Height', 'Weight', 'BMI', 'WaistC', 'HipC', 'NeckC', 'Percentage fat mass', 'Lean mass']
GAIT_PREFIXES = ('Step_', 'Stride_', 'Cadence_', 'MonoSP_', 'BiSP_', 'Speed_')

def conformal_radius(y, predictions, alpha):
    errors = np.abs(np.asarray(y) - np.asarray(predictions))
    n = len(errors)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.partition(errors, rank - 1)[rank - 1]) if rank <= n else float('inf')

def preprocess(frame, scale):
    numeric = frame.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    steps = []
    if numeric:
        numeric_steps = [('impute', SimpleImputer(strategy='median', add_indicator=True))]
        if scale:
            numeric_steps.append(('scale', StandardScaler()))
        steps.append(('num', Pipeline(numeric_steps), numeric))
    if categorical:
        steps.append(('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                       ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical))
    return ColumnTransformer(steps)

def demo(seed):
    rng = np.random.default_rng(seed)
    n = 1000
    age = rng.uniform(18, 80, n)
    return pd.DataFrame({'ID': np.arange(n), 'Age': age,
        'Height': rng.normal(170, 9, n), 'BMI': 23 + 0.055 * (age - 35) + rng.normal(0, 2.5, n),
        'WaistC': 75 + 0.21 * age + rng.normal(0, 6, n),
        'Speed_UGS': 1.8 - 0.009 * age + rng.normal(0, 0.12, n),
        'Stride_UGS': 180 - 0.6 * age + rng.normal(0, 13, n),
        'Cadence_UGS': 125 - 0.25 * age + rng.normal(0, 8, n)})

def run(args):
    if not 0 < args.alpha < 1:
        raise ValueError('--alpha must be between 0 and 1')
    if args.demo:
        df = demo(args.seed)
    elif args.csv:
        df = pd.read_csv(args.csv)
    else:
        if not (args.anthro_csv and args.gait_csv):
            raise ValueError('Provide --csv, both --anthro-csv and --gait-csv, or --demo')
        anthro = pd.read_csv(args.anthro_csv)
        gait = pd.read_csv(args.gait_csv)
        if args.id_column not in anthro or args.id_column not in gait:
            raise ValueError(f'ID column {args.id_column!r} must exist in both files')
        # Many gait datasets have repeated video rows per participant; validate anthro side.
        if anthro[args.id_column].duplicated().any():
            raise ValueError('Anthropometric CSV has duplicate IDs; deduplicate before merging')
        df = gait.merge(anthro, on=args.id_column, how='inner', validate='many_to_one')
    if args.target not in df:
        raise ValueError(f'Target {args.target!r} not in input. Available columns: {list(df.columns)}')
    df[args.target] = pd.to_numeric(df[args.target], errors='coerce')
    df = df.dropna(subset=[args.target]).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError('Need at least 30 labeled rows')
    anthro_cols = args.anthro_cols or [c for c in ANTHRO if c in df.columns]
    gait_cols = args.gait_cols or [c for c in df.columns if c.startswith(GAIT_PREFIXES)]
    anthro_cols = [c for c in anthro_cols if c != args.target and c != args.id_column]
    gait_cols = [c for c in gait_cols if c != args.target and c != args.id_column]
    if not anthro_cols or not gait_cols:
        raise ValueError('Both feature groups need columns. Specify --anthro-cols and --gait-cols if names differ.')
    if set(anthro_cols) & set(gait_cols):
        raise ValueError('Anthropometric and gait columns must not overlap')
    missing = [c for c in anthro_cols + gait_cols if c not in df]
    if missing:
        raise ValueError(f'Missing features: {missing}')
    # Split by participant when an ID is present, preventing repeated videos leaking across splits.
    indices = np.arange(len(df))
    if args.id_column in df:
        groups = df[args.id_column].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=args.seed)
        train_idx, rest_idx = next(splitter.split(df, groups=groups))
        splitter2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed + 1)
        ca, te = next(splitter2.split(df.iloc[rest_idx], groups=groups.iloc[rest_idx]))
        cal_idx, test_idx = rest_idx[ca], rest_idx[te]
    else:
        train_idx, rest_idx = train_test_split(indices, test_size=0.4, random_state=args.seed)
        cal_idx, test_idx = train_test_split(rest_idx, test_size=0.5, random_state=args.seed + 1)
    y = df[args.target]
    gait_model = Pipeline([('prep', preprocess(df[gait_cols], scale=False)),
        ('model', XGBRegressor(n_estimators=250, max_depth=3, learning_rate=0.05,
                               subsample=0.85, colsample_bytree=0.9, objective='reg:squarederror',
                               random_state=args.seed, n_jobs=2))])
    anthro_model = Pipeline([('prep', preprocess(df[anthro_cols], scale=True)),
        ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=600,
                               early_stopping=True, random_state=args.seed))])
    gait_model.fit(df.iloc[train_idx][gait_cols], y.iloc[train_idx])
    anthro_model.fit(df.iloc[train_idx][anthro_cols], y.iloc[train_idx])
    def predict(idx):
        gp = gait_model.predict(df.iloc[idx][gait_cols])
        ap = anthro_model.predict(df.iloc[idx][anthro_cols])
        return args.gait_weight * gp + (1 - args.gait_weight) * ap
    if not 0 <= args.gait_weight <= 1:
        raise ValueError('--gait-weight must be in [0,1]')
    cal_pred = predict(cal_idx)
    radius = conformal_radius(y.iloc[cal_idx], cal_pred, args.alpha)
    test_pred = predict(test_idx)
    truth = y.iloc[test_idx].to_numpy()
    lower, upper = test_pred - radius, test_pred + radius
    covered = (truth >= lower) & (truth <= upper)
    results = pd.DataFrame({'row_index': test_idx, 'true_age': truth,
        'predicted_age': test_pred, 'lower_age': lower, 'upper_age': upper, 'covered': covered})
    if args.id_column in df:
        results.insert(1, args.id_column, df.iloc[test_idx][args.id_column].to_numpy())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    metrics = {'train_rows': len(train_idx), 'calibration_rows': len(cal_idx), 'test_rows': len(test_idx),
        'anthropometric_features': anthro_cols, 'gait_features': gait_cols,
        'gait_xgboost_weight': args.gait_weight, 'anthropometric_mlp_weight': 1 - args.gait_weight,
        'target_coverage': 1 - args.alpha, 'empirical_coverage': float(covered.mean()),
        'mean_interval_width': float(np.mean(upper - lower)),
        'test_mae': float(mean_absolute_error(truth, test_pred)), 'conformal_radius': radius,
        'note': 'Split conformal marginal coverage requires exchangeable calibration/test observations. Equal weights are a fixed baseline, not optimized fusion.'}
    metrics_path = path.with_suffix('.metrics.json')
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f'Saved {path} and {metrics_path}')

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv')
    p.add_argument('--anthro-csv')
    p.add_argument('--gait-csv')
    p.add_argument('--target', default='Age')
    p.add_argument('--id-column', default='ID')
    p.add_argument('--anthro-cols', nargs='+')
    p.add_argument('--gait-cols', nargs='+')
    p.add_argument('--alpha', type=float, default=0.1)
    p.add_argument('--gait-weight', type=float, default=0.5)
    p.add_argument('--output', default='age_intervals_integrated.csv')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--demo', action='store_true')
    run(p.parse_args())
