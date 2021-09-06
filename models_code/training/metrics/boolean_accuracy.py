import torch
import torch.nn.functional as F
from typing import Optional, List
from allennlp.training.metrics import BooleanAccuracy, Metric


@Metric.register("my_boolean")
class MyBooleanAccuracy(BooleanAccuracy):
    """
    Wraps BooleanAccuracy to prep the input for sequences exact match (both tensors are of shape (batch_size, dim1))
    """

    def __init__(self, pad_id, ignore_brackets=False) -> None:
        super().__init__()
        self._pad_id = pad_id
        self._ignore_brackets = ignore_brackets

    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        ids_to_ignore: Optional[List[int]] = None
    ):
        # prep inputs: predictions size is (batch_size , num_of_pred_steps) and gold labels size is
        # (batch_size, sequence_length) where num_of_pred_steps <= max_steps and could be greater or smaller than
        # sequence_length
        sequence_length = gold_labels.size(1)
        diff = sequence_length - predictions.size(1)
        if diff > 0:
            predictions = F.pad(predictions, pad=[0, diff], value=self._pad_id)
        elif diff < 0:
            predictions = predictions[:, :sequence_length]
            mask = mask[:, :sequence_length]

        super(MyBooleanAccuracy, self).__call__(predictions, gold_labels, mask)

    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated accuracy.
        """
        if self._total_count > 0:
            accuracy = float(self._correct_count) / float(self._total_count)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return {"id_seq_acc": accuracy}
