from overrides import overrides
from typing import List

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor
from allennlp.predictors.predictor import sanitize


@Predictor.register("my_seq2seq")
class Seq2SeqPredictor(Predictor):
    """
    Predictor for sequence to sequence models, including
    [`ComposedSeq2Seq`](../models/encoder_decoders/composed_seq2seq.md) and
    [`SimpleSeq2Seq`](../models/encoder_decoders/simple_seq2seq.md) and
    [`CopyNetSeq2Seq`](../models/encoder_decoders/copynet_seq2seq.md).
    """

    def predict(self, source: str) -> JsonDict:
        return self.predict_json({"source": source})

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        outputs["source_tokens"] = instance.fields['source_tokens'].tokens
        outputs["example_id"] = instance.fields['metadata'].metadata['example_id']
        try:
            outputs["target_tokens"] = instance.fields['target_tokens'].tokens
        except KeyError:
            pass
        outputs.pop("decoder_logits", "default")
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)

        for i in range(len(instances)):
            outputs[i]["source_tokens"] = instances[i].fields['source_tokens'].tokens
            outputs[i]['example_id'] = instances[i].fields['metadata'].metadata['example_id']
            try:
                outputs[i]["target_tokens"] = instances[i].fields['target_tokens'].tokens
            except KeyError:
                pass
            outputs[i].pop("decoder_logits", "default")

        return sanitize(outputs)

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"source": "..."}`.
        """
        source = json_dict["source"]
        return self._dataset_reader.text_to_instance(source)
