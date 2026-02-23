import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.preprocessing import StandardScaler


def setup_experiment(experiment_name: str) -> str:
    """Set up MLflow experiment and return experiment_id."""
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


def extract_pipeline_params(pipeline) -> dict:
    """Extract hyperparameters from a sklearn Pipeline."""
    clf = pipeline.named_steps['clf']
    clf_type = type(clf).__name__

    params = {'scaler': 'StandardScaler', 'clf_type': clf_type}
    for k, v in clf.get_params().items():
        params[f'clf__{k}'] = v

    return params


def log_dataset_artifact(df_norm, X_train, X_test, y_train, y_test, label_classes) -> None:
    """Log a compact dataset summary CSV as an artifact (not the full CSV)."""
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total = X_train.shape[0] + X_test.shape[0]

    rows = [
        {'metric': 'n_train_samples', 'value': X_train.shape[0]},
        {'metric': 'n_test_samples',  'value': X_test.shape[0]},
        {'metric': 'n_features',      'value': X_train.shape[1]},
        {'metric': 'n_classes',       'value': len(label_classes)},
        {'metric': 'train_test_split',
         'value': f'{X_train.shape[0]/total:.0%}/{X_test.shape[0]/total:.0%}'},
    ]
    for idx, cls in enumerate(label_classes):
        rows.append({'metric': f'train_class_{cls}', 'value': int(class_counts.get(idx, 0))})

    summary_df = pd.DataFrame(rows)

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    try:
        summary_df.to_csv(tmp.name, index=False)
        tmp.close()
        mlflow.log_artifact(tmp.name, artifact_path='dataset')
    finally:
        os.unlink(tmp.name)


def log_model_run(model_name, pipeline, metrics, X_train, X_test,
                  y_train, y_test, df_norm, label_classes, is_best: bool) -> str:
    """Open an MLflow run, log params/metrics/artifacts/model, return run_id."""
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.log_params(extract_pipeline_params(pipeline))

        metric_map = {
            'Accuracy':  'accuracy',
            'Precision': 'precision',
            'Recall':    'recall',
            'F1-Score':  'f1_score',
        }
        mlflow.log_metrics({metric_map[k]: v for k, v in metrics.items()})

        log_dataset_artifact(df_norm, X_train, X_test, y_train, y_test, label_classes)

        artifact_path = (
            model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
        )
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path=artifact_path,
            input_example=X_train[:5],
        )

        mlflow.set_tag('best_model', str(is_best))
        mlflow.set_tag('model_name', model_name)

        return run.info.run_id


def log_comparison_chart(results: dict, run_ids: dict, best_model_name: str) -> None:
    """Grouped bar chart of all metrics; log to every run under charts/."""
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = list(results.keys())
    n_models = len(model_names)

    x = np.arange(len(metric_keys))
    width = 0.8 / n_models
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [results[name][m] for m in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, alpha=0.85)
        if name == best_model_name:
            for bar in bars:
                bar.set_edgecolor('gold')
                bar.set_linewidth(2.5)

    ax.set_xlabel('Metric')
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — hand-gesture-classification', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0.88, 1.01)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    try:
        tmp.close()
        fig.savefig(tmp.name, dpi=150, bbox_inches='tight')
        plt.close(fig)
        client = MlflowClient()
        for run_id in run_ids.values():
            client.log_artifact(run_id, tmp.name, artifact_path='charts')

        # client.set_experiment_tag(run_id, 'status', 'completed')
        # client.set_experiment_tag(run_id, 'owner', 'Omar Gamal')
    finally:
        os.unlink(tmp.name)


def tag_best_model_run(run_ids: dict, results: dict, metric_key: str = 'F1-Score') -> str:
    """Tag best_model=True/False on all completed runs; return best model name."""
    best_model_name = max(results, key=lambda n: results[n][metric_key])

    client = MlflowClient()
    for model_name, run_id in run_ids.items():
        client.set_tag(run_id, 'best_model', str(model_name == best_model_name))
        client.set_tag(run_id, 'model_name', model_name)

    return best_model_name


def register_best_model(
    run_ids: dict,
    results: dict,
    registered_model_name: str = 'hand-gesture-classifier',
    metric_key: str = 'F1-Score',
) -> str:
    """Register the best model in the MLflow Model Registry, alias it 'champion', return version."""
    best_model_name = max(results, key=lambda n: results[n][metric_key])
    best_run_id = run_ids[best_model_name]

    artifact_path = (
        best_model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    )
    model_uri = f'runs:/{best_run_id}/{artifact_path}'

    mv = mlflow.register_model(model_uri=model_uri, name=registered_model_name)

    client = MlflowClient()
    client.set_registered_model_alias(registered_model_name, 'champion', mv.version)

    f1 = results[best_model_name][metric_key]
    client.update_model_version(
        name=registered_model_name,
        version=mv.version,
        description=(
            f'Best model: {best_model_name}. '
            f'{metric_key}: {f1:.4f}. '
            f'Experiment: hand-gesture-classification.'
        ),
    )

    print(f'  Registered "{registered_model_name}" v{mv.version} '
          f'(alias: champion) ← {best_model_name}')
    return mv.version

def run_mlflow_tracking(
    models: dict,
    results: dict,
    trained_models: dict,
    X_train,
    X_test,
    y_train,
    y_test,
    df_norm,
    label_classes,
    experiment_name: str = 'hand-gesture-classification',
) -> dict:
    """Orchestrator: log all 4 model runs and return {model_name: run_id}."""
    experiment_id = setup_experiment(experiment_name)

    client = MlflowClient()
    client.set_experiment_tag(experiment_id, 'owner', 'Omar Gamal')

    best_model_name = max(results, key=lambda n: results[n]['F1-Score'])

    run_ids = {}
    for model_name in models:
        run_id = log_model_run(
            model_name=model_name,
            pipeline=trained_models[model_name],
            metrics=results[model_name],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            df_norm=df_norm,
            label_classes=label_classes,
            is_best=(model_name == best_model_name),
        )
        run_ids[model_name] = run_id
        print(f'  Logged {model_name}: {run_id}')

    
    tag_best_model_run(run_ids, results)
    log_comparison_chart(results, run_ids, best_model_name)
    register_best_model(run_ids, results)


    return run_ids
