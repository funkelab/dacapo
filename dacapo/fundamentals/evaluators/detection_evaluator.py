from .helpers import Evaluator

import funlib.evaluate
import daisy

import attr
import numpy as np

from typing import Optional


@attr.s
class DetectionEvaluator(Evaluator):
    return_results: bool = attr.ib(default=False)
    background_label: Optional[int] = attr.ib(default=None)
    matching_score: str = attr.ib(default="overlap")
    matching_threshold: float = attr.ib(default=1.0)

    def evaluate_block(self, predicted, ground_truth):
        gt_label_data = ground_truth.to_ndarray(roi=predicted.roi)
        pred_label_data = predicted.to_ndarray(roi=predicted.roi)

        # PIXEL-WISE SCORES

        sample_scores = {}

        # accuracy

        sample_scores["accuracy"] = (
            (pred_label_data != self.background_label) == (gt_label_data != self.background_label)
        ).sum() / gt_label_data.size
        fg_mask = gt_label_data != self.background_label
        sample_scores["fg_accuracy"] = (
            (pred_label_data[fg_mask] != self.background_label)
            == (gt_label_data[fg_mask] != self.background_label)
        ).sum() / fg_mask.sum()

        # precision, recall, fscore

        relevant = gt_label_data != self.background_label
        selected = pred_label_data != self.background_label
        num_relevant = relevant.sum()
        num_selected = selected.sum()

        tp = (pred_label_data[relevant] != self.background_label).sum()
        fp = num_selected - tp
        tn = (gt_label_data.size - num_relevant) - fp

        # precision, or positive predictive value
        if num_selected > 0:
            ppv = tp / num_selected  # = tp/(tp + fp)
        else:
            ppv = np.nan
        # recall, or true positive rate
        if num_relevant > 0:
            tpr = tp / num_relevant  # = tp/(tp + fn)
        else:
            tpr = np.nan
        # specificity, or true negative rate
        if tn + fp > 0:
            tnr = tn / (tn + fp)
        else:
            tnr = np.nan
        # fall-out, or false positive rate
        if tn + fp > 0:
            fpr = fp / (tn + fp)
        else:
            fpr = np.nan

        if ppv + tpr > 0:
            fscore = 2 * (ppv * tpr) / (ppv + tpr)
        else:
            fscore = np.nan
        balanced_accuracy = (tpr + tnr) / 2

        sample_scores["ppv"] = ppv
        sample_scores["tpr"] = tpr
        sample_scores["tnr"] = tnr
        sample_scores["fpr"] = fpr
        sample_scores["fscore"] = fscore
        sample_scores["balanced_accuracy"] = balanced_accuracy

        # DETECTION SCORES (on foreground objects only)

        detection_scores = funlib.evaluate.detection_scores(
            gt_label_data,
            pred_label_data,
            matching_score=self.matching_score,
            matching_threshold=self.matching_threshold,
            voxel_size=predicted.voxel_size,
            return_matches=True,
        )
        components = {}
        tp = detection_scores["tp"]
        fp = detection_scores["fp"]
        fn = detection_scores["fn"]
        num_selected = tp + fp
        num_relevant = tp + fn

        # precision, or positive predictive value
        if num_selected > 0:
            ppv = tp / num_selected  # = tp/(tp + fp)
        else:
            ppv = np.nan
        # recall, or true positive rate
        if num_relevant > 0:
            tpr = tp / num_relevant  # = tp/(tp + fn)
        else:
            tpr = np.nan

        if ppv + tpr > 0:
            fscore = 2 * (ppv * tpr) / (ppv + tpr)
        else:
            fscore = np.nan

        sample_scores["detection_ppv"] = ppv
        sample_scores["detection_tpr"] = tpr
        sample_scores["detection_fscore"] = fscore
        sample_scores["avg_iou"] = detection_scores["avg_iou"]

        if self.return_results:

            components_gt = detection_scores["components_truth"]
            components_pred = detection_scores["components_test"]
            matches = detection_scores["matches"]
            matches_gt = np.array([m[1] for m in matches])
            matches_pred = np.array([m[0] for m in matches])
            components_tp_gt = np.copy(components_gt)
            components_tp_pred = np.copy(components_pred)
            components_fn_gt = np.copy(components_gt)
            components_fp_pred = np.copy(components_pred)
            tp_gt_mask = np.isin(components_gt, matches_gt)
            tp_pred_mask = np.isin(components_pred, matches_pred)
            components_tp_gt[np.logical_not(tp_gt_mask)] = 0
            components_tp_pred[np.logical_not(tp_pred_mask)] = 0
            components_fn_gt[tp_gt_mask] = 0
            components_fp_pred[tp_pred_mask] = 0

            components["volumes/components_tp_gt"] = daisy.Array(
                components_tp_gt, predicted.roi, predicted.voxel_size
            )
            components["volumes/components_fn_gt"] = daisy.Array(
                components_fn_gt, predicted.roi, predicted.voxel_size
            )
            components["volumes/components_tp_pred"] = daisy.Array(
                components_tp_pred, predicted.roi, predicted.voxel_size
            )
            components["volumes/components_fp_pred"] = daisy.Array(
                components_fp_pred, predicted.roi, predicted.voxel_size
            )

        scores = {"sample": sample_scores, "average": sample_scores}

        if self.return_results:
            components.update(
                {
                    "volumes/predicted": daisy.Array(
                        predicted.data.astype(np.uint64),
                        roi=predicted.roi,
                        voxel_size=predicted.voxel_size,
                    ),
                    "volumes/ground_truth": daisy.Array(
                        ground_truth.data.astype(np.uint64),
                        roi=ground_truth.roi,
                        voxel_size=ground_truth.voxel_size,
                    ),
                }
            )
            return scores, components

        return scores
