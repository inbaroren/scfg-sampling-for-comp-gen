import os
from allennlp.training.trainer import BatchCallback, EpochCallback
from typing import Any, Dict,  List
from allennlp.common import FromParams
from comet_ml import Experiment, ExistingExperiment
from allennlp.common.params import Params
import socket


class CometMLExperiment(FromParams):
    def __init__(self, api_key, project_name):
        if "COMET_EXPERIMENT_KEY" in os.environ:
            obj = ExistingExperiment(
                api_key=api_key,
                project_name=project_name,
                previous_experiment=os.environ["COMET_EXPERIMENT_KEY"],
                auto_weight_logging=True,
            )
        else:
            obj = Experiment(api_key=api_key, project_name=project_name)
            os.environ["COMET_EXPERIMENT_KEY"] = obj.get_key()

        class MockSummaryWriter:
            def __init__(self, is_train, prev_value):
                self.is_train = is_train
                self.prev_value = prev_value
                if is_train:
                    self.cntx = obj.train
                else:
                    self.cntx = obj.validate

            def add_scalar(self, name, value, timestep):
                if 'bias' in name:
                    return
                name = name.replace('token_embedder_tokens.transformer_model', '')
                name = name.replace('encoder', 'enc')
                name = name.replace('decoder', 'dec')
                with self.cntx():
                    # if not ('bias' in name):
                    obj.log_metric(name=name, value=value, step=timestep)

            def add_histogram(self, name, value, timestep):
                if 'bias' in name:
                    return
                name = name.replace('token_embedder_tokens.transformer_model', '')
                name = name.replace('encoder', 'enc')
                name = name.replace('decoder', 'dec')
                with self.cntx():
                    obj.log_histogram_3d(name=name, value=value, step=timestep)

            def close(self):
                self.prev_value.close()

        self.obj = obj
        self.mock_obj = MockSummaryWriter


@BatchCallback.register("prediction_logger")
class PredictionLogger(BatchCallback):
    def __init__(self, experiment: CometMLExperiment):
        super().__init__()
        self.experiment = experiment.obj
        self.mock_obj = experiment.mock_obj

    def __call__(
            self,
            trainer: "GradientDescentTrainer",
            batch_inputs,
            batch_outputs: List[Dict[str, Any]],
            epoch: int,
            batch_number: int,
            is_training: bool,
            is_master: bool,
    ) -> None:
        if not is_master:
            return
        if not is_training:
            return
        # pass
        with self.experiment.train():
            self.experiment.log_metrics(trainer.model.get_metrics(), epoch=epoch, step=batch_number*epoch)


@EpochCallback.register("performance_logger")
class PerformanceLogger(EpochCallback):
    def __init__(self, experiment: CometMLExperiment):
        super().__init__()
        self.experiment = experiment.obj
        self.mock_obj = experiment.mock_obj

    def __call__(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
    ) -> None:
        if not is_master:
            return
        if epoch == -1:
            self.experiment.log_other("hostname", socket.gethostname())
            self.experiment.set_name(trainer.model._experiment_name)
            params = Params.from_file(f"{trainer._serialization_dir}/config.json").as_flat_dict()
            self.experiment.log_parameters(params)
            return
        self.experiment.log_metrics(metrics, epoch=epoch)
        self.experiment.log_epoch_end(epoch, step=None)


@EpochCallback.register("end_logger")
class EndLogger(EpochCallback):
    def __init__(self, experiment: CometMLExperiment):
        super().__init__()
        self.experiment = experiment.obj
        self.mock_obj = experiment.mock_obj

    def __call__(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any],
            epoch: int,
            is_master: bool,
    ) -> None:
        if not is_master:
            return
        self._experiment.add_tag("COMPLETED")