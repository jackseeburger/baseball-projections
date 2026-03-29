"""Weights & Biases tracking helpers for PyMC baseball models.

Handles experiment logging, MCMC diagnostics, posterior visualization,
accuracy metrics, and model artifact management.

Usage:
    tracker = WandbTracker(
        run_name="hitter-v1-2026",
        model_type="hitter",
        config={"n_samples": 2000, "n_chains": 4},
    )
    tracker.log_mcmc_diagnostics(trace)
    tracker.log_accuracy_vs_marcel(projections, marcel, actuals)
    tracker.log_posterior_plots(trace, params=["mu_ba", "mu_obp"])
    tracker.save_model_artifact(trace, metadata={...})
    tracker.finish()
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

WANDB_PROJECT = "baseball-projections"
WANDB_ENTITY = "jseeburger"  # your wandb entity


class WandbTracker:
    """Wrapper around wandb for baseball projection experiments."""

    def __init__(
        self,
        run_name: str,
        model_type: str = "hitter",
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        notes: str = "",
        offline: bool = False,
    ):
        """Initialize a wandb run.

        Args:
            run_name: Name for this run (e.g., "hitter-v1-2026").
            model_type: "hitter" or "pitcher".
            config: Hyperparameters and settings to log.
            tags: Tags for organizing runs.
            notes: Free-text notes about this run.
            offline: If True, log locally (sync later with `wandb sync`).
        """
        import wandb

        if offline:
            os.environ["WANDB_MODE"] = "offline"

        default_tags = [model_type]
        all_tags = default_tags + (tags or [])

        self.run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=config or {},
            tags=all_tags,
            notes=notes,
            group=model_type,
            job_type="train",
            reinit=True,
        )
        self._wandb = wandb
        self.run_name = run_name

    # ═══════════════════════════════════════════════════════════════════════
    # MCMC Diagnostics
    # ═══════════════════════════════════════════════════════════════════════

    def log_mcmc_diagnostics(self, trace) -> dict:
        """Log MCMC diagnostics from an ArviZ InferenceData trace.

        Logs:
            - R-hat (Gelman-Rubin) per parameter
            - Effective sample size (ESS) bulk and tail
            - Number of divergences
            - Summary statistics

        Returns dict of diagnostics for programmatic use.
        """
        import arviz as az

        diagnostics = {}

        # R-hat
        rhat = az.rhat(trace)
        rhat_vals = {}
        for var in rhat.data_vars:
            vals = rhat[var].values
            if vals.ndim == 0:
                rhat_vals[var] = float(vals)
            else:
                rhat_vals[f"{var}_max"] = float(np.max(vals))
                rhat_vals[f"{var}_mean"] = float(np.mean(vals))
        diagnostics["rhat"] = rhat_vals
        self._wandb.log({f"diagnostics/rhat/{k}": v for k, v in rhat_vals.items()})

        # ESS
        ess_bulk = az.ess(trace, method="bulk")
        ess_tail = az.ess(trace, method="tail")
        ess_vals = {}
        for var in ess_bulk.data_vars:
            bulk = ess_bulk[var].values
            tail = ess_tail[var].values
            if bulk.ndim == 0:
                ess_vals[f"{var}_bulk"] = float(bulk)
                ess_vals[f"{var}_tail"] = float(tail)
            else:
                ess_vals[f"{var}_bulk_min"] = float(np.min(bulk))
                ess_vals[f"{var}_tail_min"] = float(np.min(tail))
        diagnostics["ess"] = ess_vals
        self._wandb.log({f"diagnostics/ess/{k}": v for k, v in ess_vals.items()})

        # Divergences
        if hasattr(trace, "sample_stats"):
            div = trace.sample_stats.get("diverging")
            if div is not None:
                n_div = int(div.values.sum())
            else:
                n_div = 0
        else:
            n_div = 0
        diagnostics["divergences"] = n_div
        self._wandb.log({"diagnostics/divergences": n_div})

        # Overall health check
        max_rhat = max(rhat_vals.values()) if rhat_vals else 0.0
        min_ess = min(v for k, v in ess_vals.items() if "bulk" in k) if ess_vals else 0
        diagnostics["healthy"] = max_rhat < 1.05 and n_div == 0 and min_ess > 400
        self._wandb.log({"diagnostics/healthy": diagnostics["healthy"]})

        # Summary table
        summary = az.summary(trace)
        table = self._wandb.Table(dataframe=summary.reset_index())
        self._wandb.log({"diagnostics/summary": table})

        return diagnostics

    # ═══════════════════════════════════════════════════════════════════════
    # Posterior Visualization
    # ═══════════════════════════════════════════════════════════════════════

    def log_posterior_plots(
        self,
        trace,
        params: Optional[list[str]] = None,
        prefix: str = "posterior",
    ):
        """Log posterior distribution plots as wandb images.

        Args:
            trace: ArviZ InferenceData.
            params: Specific parameter names to plot (None = all).
            prefix: Metric prefix for grouping.
        """
        import arviz as az
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Trace plot
        ax = az.plot_trace(trace, var_names=params, compact=True)
        fig = ax.ravel()[0].figure
        self._wandb.log({f"{prefix}/trace_plot": self._wandb.Image(fig)})
        plt.close(fig)

        # Forest plot
        try:
            ax = az.plot_forest(trace, var_names=params, combined=True)
            fig = ax.ravel()[0].figure
            self._wandb.log({f"{prefix}/forest_plot": self._wandb.Image(fig)})
            plt.close(fig)
        except Exception:
            pass  # forest plot can fail with single-dim params

        # Posterior plot
        ax = az.plot_posterior(trace, var_names=params)
        if hasattr(ax, "ravel"):
            fig = ax.ravel()[0].figure
        else:
            fig = ax.figure
        self._wandb.log({f"{prefix}/posterior_plot": self._wandb.Image(fig)})
        plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # Accuracy Metrics
    # ═══════════════════════════════════════════════════════════════════════

    def log_accuracy_metrics(
        self,
        projections: "pd.DataFrame",
        actuals: "pd.DataFrame",
        stats: list[str] = None,
        prefix: str = "accuracy",
    ):
        """Log projection accuracy (MAE, RMSE, correlation) for key stats.

        Args:
            projections: DataFrame with player_id + stat columns.
            actuals: DataFrame with player_id + stat columns (same schema).
            stats: Which stat columns to evaluate. Defaults to batting stats.
            prefix: Metric prefix.
        """
        import pandas as pd

        if stats is None:
            stats = ["ba", "obp", "slg", "woba", "wrc_plus"]

        # Merge on player_id
        merged = projections.merge(
            actuals, on="player_id", suffixes=("_proj", "_actual")
        )

        metrics = {}
        for stat in stats:
            proj_col = f"{stat}_proj"
            act_col = f"{stat}_actual"
            if proj_col not in merged.columns or act_col not in merged.columns:
                continue

            diff = merged[proj_col] - merged[act_col]
            mae = float(diff.abs().mean())
            rmse = float(np.sqrt((diff ** 2).mean()))
            corr = float(merged[proj_col].corr(merged[act_col]))

            metrics[f"{prefix}/{stat}/mae"] = mae
            metrics[f"{prefix}/{stat}/rmse"] = rmse
            metrics[f"{prefix}/{stat}/correlation"] = corr

        self._wandb.log(metrics)
        return metrics

    def log_accuracy_vs_marcel(
        self,
        projections: "pd.DataFrame",
        marcel: "pd.DataFrame",
        actuals: "pd.DataFrame",
        stats: list[str] = None,
        prefix: str = "vs_marcel",
    ):
        """Compare Bayesian model accuracy against Marcel baseline.

        Logs side-by-side MAE/RMSE/correlation for both systems.
        """
        import pandas as pd

        if stats is None:
            stats = ["ba", "obp", "slg", "woba"]

        bayes_metrics = self.log_accuracy_metrics(
            projections, actuals, stats, prefix=f"{prefix}/bayesian"
        )
        marcel_metrics = self.log_accuracy_metrics(
            marcel, actuals, stats, prefix=f"{prefix}/marcel"
        )

        # Log improvement
        for stat in stats:
            b_mae = bayes_metrics.get(f"{prefix}/bayesian/{stat}/mae")
            m_mae = marcel_metrics.get(f"{prefix}/marcel/{stat}/mae")
            if b_mae is not None and m_mae is not None and m_mae > 0:
                improvement = (m_mae - b_mae) / m_mae * 100
                self._wandb.log({f"{prefix}/improvement/{stat}_mae_pct": improvement})

        return {"bayesian": bayes_metrics, "marcel": marcel_metrics}

    # ═══════════════════════════════════════════════════════════════════════
    # Training Progress
    # ═══════════════════════════════════════════════════════════════════════

    def log_training_step(self, step: int, metrics: dict):
        """Log metrics during iterative training / sampling.

        Args:
            step: Current step number.
            metrics: Dict of metric_name -> value.
        """
        self._wandb.log({"step": step, **metrics})

    def log_config_update(self, updates: dict):
        """Update run config with additional parameters."""
        self.run.config.update(updates)

    # ═══════════════════════════════════════════════════════════════════════
    # Artifacts
    # ═══════════════════════════════════════════════════════════════════════

    def save_model_artifact(
        self,
        trace,
        metadata: Optional[dict] = None,
        artifact_name: Optional[str] = None,
        aliases: Optional[list[str]] = None,
    ):
        """Save a PyMC trace as a wandb artifact.

        Args:
            trace: ArviZ InferenceData object.
            metadata: Extra metadata to attach.
            artifact_name: Name for the artifact. Defaults to run_name.
            aliases: Artifact aliases (e.g., ["latest", "best"]).
        """
        import arviz as az

        name = artifact_name or self.run_name.replace(" ", "-")
        artifact = self._wandb.Artifact(
            name=name,
            type="model",
            metadata=metadata or {},
        )

        # Save trace as NetCDF
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = os.path.join(tmpdir, "trace.nc")
            trace.to_netcdf(trace_path)
            artifact.add_file(trace_path, name="trace.nc")

            # Also save metadata JSON
            if metadata:
                meta_path = os.path.join(tmpdir, "metadata.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
                artifact.add_file(meta_path, name="metadata.json")

        self._wandb.log_artifact(artifact, aliases=aliases or ["latest"])

    def save_projections_artifact(
        self,
        projections: "pd.DataFrame",
        artifact_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Save projection results as a wandb artifact (parquet).

        Args:
            projections: DataFrame of player projections.
            artifact_name: Artifact name.
            metadata: Extra metadata.
        """
        name = artifact_name or f"{self.run_name}-projections"
        artifact = self._wandb.Artifact(
            name=name,
            type="projections",
            metadata=metadata or {},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "projections.parquet")
            projections.to_parquet(path, index=False)
            artifact.add_file(path, name="projections.parquet")

            # Also log as a wandb Table for interactive viewing
            table = self._wandb.Table(dataframe=projections.head(500))
            self._wandb.log({"projections_preview": table})

        self._wandb.log_artifact(artifact, aliases=["latest"])

    # ═══════════════════════════════════════════════════════════════════════
    # Data Logging
    # ═══════════════════════════════════════════════════════════════════════

    def log_dataset_stats(
        self,
        df: "pd.DataFrame",
        name: str = "training_data",
    ):
        """Log dataset summary statistics.

        Args:
            df: The dataset DataFrame.
            name: Name prefix for metrics.
        """
        self._wandb.log({
            f"{name}/n_rows": len(df),
            f"{name}/n_cols": len(df.columns),
            f"{name}/columns": list(df.columns),
        })

        # Log numeric column distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:20]:  # cap at 20 to avoid flooding
            self._wandb.log({
                f"{name}/stats/{col}/mean": float(df[col].mean()),
                f"{name}/stats/{col}/std": float(df[col].std()),
                f"{name}/stats/{col}/min": float(df[col].min()),
                f"{name}/stats/{col}/max": float(df[col].max()),
                f"{name}/stats/{col}/null_pct": float(df[col].isna().mean()),
            })

    # ═══════════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def finish(self, quiet: bool = False):
        """Finish the wandb run.

        Args:
            quiet: If True, suppress the run summary output.
        """
        self._wandb.finish(quiet=quiet)

    @property
    def url(self) -> str:
        """URL to the wandb run dashboard."""
        return self.run.url

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish(quiet=True)
