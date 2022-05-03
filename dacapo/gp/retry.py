import copy
import logging

from gunpowder.nodes.batch_provider import BatchRequestError
from gunpowder.profiling import Timing

import gunpowder as gp

logger = logging.getLogger(__name__)


class Retry(gp.BatchFilter):
    """A Gunpowder node for retrying a batch request."""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts

    def provide(self, request):

        timing_prepare = Timing(self, "prepare")
        timing_prepare.start()

        upstream_request = request.copy()

        timing_prepare.stop()

        batch = None
        for i in range(self.max_attempts):
            try:
                upstream_request._random_seed += 1
                batch = self.get_upstream_provider().request_batch(upstream_request)
                break
            except BatchRequestError as e:
                if i + 1 < self.max_attempts:
                    continue
                else:
                    raise BatchRequestError(
                        f"Could not get a valid batch in {self.max_attempts} attempts"
                    )

        return batch
