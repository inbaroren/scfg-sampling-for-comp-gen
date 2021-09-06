from overrides import overrides
from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.predictor import sanitize


@Predictor.register("my_seq2seq_para_filter")
class Seq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    [`ComposedSeq2Seq`](../models/encoder_decoders/composed_seq2seq.md) and
    [`SimpleSeq2Seq`](../models/encoder_decoders/simple_seq2seq.md) and
    [`CopyNetSeq2Seq`](../models/encoder_decoders/copynet_seq2seq.md).
    I use it to evaluate millions of instances, so I save only an evaluation indicator for each example
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs["correct"] = [t.text for t in instance.fields['target_tokens'].tokens[1:-1]] == \
                             [t.text for t in outputs['predicted_tokens'][:-1]]
        outputs["example_id"] = instance.fields['metadata'].metadata['example_id']
        outputs.pop("decoder_logits", "default")
        outputs.pop("predicted_tokens", "default")
        outputs.pop("predictions", "default")
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        for i in range(len(instances)):
            outputs[i]["correct"] = [t.text for t in instances[i].fields['target_tokens'].tokens[1:-1]] == \
                                    [t.text for t in outputs[i]['predicted_tokens'][:-1]]
            outputs[i]['example_id'] = instances[i].fields['metadata'].metadata['example_id']
            outputs[i].pop("decoder_logits", "default")
            outputs[i].pop("predicted_tokens", "default")
            outputs[i].pop("predictions", "default")

        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
