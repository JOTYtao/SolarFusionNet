import torch
import wandb
from torchmetrics import MeanMetric
import os
from pl_models.utils import ContextMixerModule
import numpy as np

class FusionFormer(ContextMixerModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        metrics: dict,
        criterion: torch.nn.Module,
        **kwargs,
    ):
        super().__init__(model, optimizer, scheduler, metrics, criterion, kwargs=kwargs)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True):
        out, bands_weights, feature_weights, attn_output_weights = self.model(
            x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask
        )
        return out, bands_weights, feature_weights, attn_output_weights

    def training_step(self, train_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(train_batch)

        y_hat, bands_weights, feature_weights, attn_output_weights = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=True)
        # y_hat = y_hat.mean(dim=2)

        loss = self.criterion(y_hat, y_ts)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.train:
            metric = getattr(self, f"train_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"train/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        return loss

    def validation_step(self, val_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(val_batch)

        y_hat, bands_weights, feature_weights, attn_output_weights = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)
        # y_hat = y_hat.mean(dim=2)


        loss = self.criterion(y_hat, y_ts)

        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        return {"predictions": y_hat, "ground_truth": y_ts}


    def test_step(self, test_batch, batch_idx):
        (
            x_ts,
            x_ctx,
            y_ts,
            y_prev_ts,
            ctx_coords,
            ts_coords,
            time_coords,
        ) = self.prepare_batch(test_batch)
        y_hat, bands_weights, feature_weights, attn_output_weights = self(x_ctx, ctx_coords, x_ts, ts_coords, time_coords, mask=False)

        loss = self.criterion(y_hat, y_ts)

        self.val_loss(loss)
        self.log("test/loss", self.val_loss, on_step=True, prog_bar=True)

        for key in self.hparams.metrics.val:
            metric = getattr(self, f"val_{key}")
            if hasattr(metric, "needs_previous") and metric.needs_previous:
                metric(y_hat, y_ts, y_prev_ts)
            else:
                metric(y_hat, y_ts)
            self.log(
                f"val/{key}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        # return {"predictions": y_hat, "ground_truth": y_ts}

        return {"predictions": y_hat,
                "ground_truth": y_ts,
                "bands_weights": bands_weights,
                "feature_weights": feature_weights,
                "attn_output_weights": attn_output_weights,
                "time_coords": time_coords
            }

    def test_epoch_end(self, outputs):

        save_dir = ''
        os.makedirs(save_dir, exist_ok=True)
        all_predictions, all_ground_truths, all_attns, all_bands_weights, all_sat_weights, all_feat_weights, all_attn_out_weights, all_time_coords = [], [], [], [], [], [], [], []
        for batch_outputs in outputs:
            all_predictions.append(batch_outputs['predictions'].cpu().numpy())
            all_ground_truths.append(batch_outputs['ground_truth'].cpu().numpy())
            all_time_coords.append(batch_outputs['time_coords'].cpu().numpy())
        np.save(os.path.join(save_dir, 'all_predictions.npy'), np.concatenate(all_predictions, axis=0))
        np.save(os.path.join(save_dir, 'all_ground_truths.npy'), np.concatenate(all_ground_truths, axis=0))

        np.save(os.path.join(save_dir, 'all_time_coords.npy'), np.concatenate(all_time_coords, axis=0))
