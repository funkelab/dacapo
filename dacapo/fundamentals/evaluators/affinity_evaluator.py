from .helpers import Evaluator

from funlib.evaluate import rand_voi
import daisy

import attr
import numpy as np


@attr.s
class AffinityEvaluator(Evaluator):
    return_results: bool = attr.ib(default=False)

    def evaluate(self, predicted, ground_truth):
        gt_label_data = ground_truth.to_ndarray(roi=predicted.roi)
        pred_label_data = predicted.to_ndarray(roi=predicted.roi)

        results = rand_voi(gt_label_data, pred_label_data)
        results["voi_sum"] = results["voi_split"] + results["voi_merge"]

        scores = {"sample": results, "average": results}

        if self.return_results:
            results = {
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

            return scores, results

        return scores
