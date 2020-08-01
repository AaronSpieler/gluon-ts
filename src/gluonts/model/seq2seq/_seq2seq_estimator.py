# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# Standard library imports
import logging
from distutils.util import strtobool
from typing import List, Optional

# Third-party imports
import mxnet as mx
import numpy as np

# First-party imports
from gluonts import transform
from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.stat import calculate_dataset_statistics
from gluonts.model.estimator import GluonEstimator
from gluonts.model.forecast import Quantile
from gluonts.model.forecast_generator import QuantileForecastGenerator
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor
from gluonts.mx.block.decoder import OneShotDecoder, Seq2SeqDecoder
from gluonts.mx.block.enc2dec import (
    PassThroughEnc2Dec,
    FutureFeatIntegratorEnc2Dec,
)
from gluonts.mx.block.encoder import (
    HierarchicalCausalConv1DEncoder,
    MLPEncoder,
    RNNEncoder,
    Seq2SeqEncoder,
)
from gluonts.mx.block.feature import FeatureEmbedder
from gluonts.mx.block.quantile_output import QuantileOutput
from gluonts.mx.block.scaler import NOPScaler, Scaler
from gluonts.mx.trainer import Trainer
from gluonts.support.util import copy_parameters
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.transform import (
    ExpectedNumInstanceSampler,
    AddTimeFeatures,
    Transformation,
    AddObservedValuesIndicator,
    RemoveFields,
    AddAgeFeature,
    RenameFields,
    AddConstFeature,
    VstackFeatures,
    SetField,
    InstanceSplitter,
)

# Relative imports
from gluonts.transform.field import FuturePadField, ConcatFields

from ._seq2seq_network import Seq2SeqPredictionNetwork, Seq2SeqTrainingNetwork


class Seq2SeqEstimator(GluonEstimator):
    """
    Quantile-Regression Sequence-to-Sequence Estimator

    """

    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        encoder: Seq2SeqEncoder,
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        decoder: Seq2SeqDecoder = None,
        cardinality: List[int] = None,
        embedding_dimension: List[int] = None,
        scaler: Scaler = NOPScaler(),
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        trainer: Trainer = Trainer(),
        num_parallel_samples: int = 100,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        use_feat_dynamic_real: bool = False,
        use_past_feat_dynamic_real: bool = False,
        use_feat_static_cat: bool = False,
    ) -> None:
        assert (
            prediction_length > 0
        ), "The value of `prediction_length` should be > 0"
        assert (
            context_length is None or context_length > 0
        ), "The value of `context_length` should be > 0"
        assert quantiles is None or all(
            0 <= d <= 1 for d in quantiles
        ), "Elements of `quantiles` should be >= 0 and <= 1"

        super().__init__(trainer=trainer)

        self.context_length = (
            context_length
            if context_length is not None
            else prediction_length * 4
        )
        self.prediction_length = prediction_length
        self.freq = freq
        self.quantiles = (
            quantiles
            if quantiles is not None
            else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_mlp_layer = decoder_mlp_layer
        self.decoder_mlp_static_dim = decoder_mlp_static_dim
        self.scaler = scaler
        self.cardinality = (
            cardinality if cardinality and use_feat_static_cat else [1]
        )
        self.embedding_dimension = (
            embedding_dimension
            if embedding_dimension is not None
            else [min(50, (cat + 1) // 2) for cat in self.cardinality]
        )
        self.embedder = FeatureEmbedder(
            cardinalities=self.cardinality,
            embedding_dims=self.embedding_dimension,
            # dtype=np.float32,
        )
        self.num_parallel_samples = num_parallel_samples

        self.add_time_feature = add_time_feature
        self.add_age_feature = add_age_feature
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.use_past_feat_dynamic_real = use_past_feat_dynamic_real
        self.use_feat_static_cat = use_feat_static_cat

    def create_transformation(self) -> Transformation:
        chain = []
        dynamic_feat_fields = []

        # BACKWARDS COMPATIBILITY

        chain.append(
            RenameFields({"dynamic_feat": FieldName.FEAT_DYNAMIC_REAL})
        )

        # REMOVE SLACK

        remove_field_names = [
            FieldName.FEAT_DYNAMIC_CAT,
            FieldName.FEAT_STATIC_REAL,
        ]

        if not self.use_past_feat_dynamic_real:
            remove_field_names.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        else:
            dynamic_feat_fields.append(FieldName.FEAT_DYNAMIC_REAL)

        if not self.use_feat_static_cat:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)

        chain.append(RemoveFields(field_names=remove_field_names))

        # ADD INDICATOR SEQUENCES

        chain.append(
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
                dtype=self.dtype,
            ),
        )

        if self.add_age_feature:
            chain.append(
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.prediction_length,
                    dtype=self.dtype,
                ),
            )
            dynamic_feat_fields.append(FieldName.FEAT_AGE)

        if self.add_time_feature:
            chain.append(
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(self.freq),
                    pred_length=self.prediction_length,
                ),
            )
            dynamic_feat_fields.append(FieldName.FEAT_TIME)

        # PROCESS STATIC CAT

        if not self.use_feat_static_cat:
            chain.append(
                SetField(
                    output_field=FieldName.FEAT_STATIC_CAT,
                    value=np.array([0.0]),
                ),
            )

        # SPLIT THE TARGET SEQUENCE
        # PAD PAST TARGET, PAST OBSERVED AND PAST DYNAMIC FEAT

        if self.use_past_feat_dynamic_real:
            chain.append(
                FuturePadField(
                    output_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    target_field=FieldName.TARGET,
                    input_field=FieldName.PAST_FEAT_DYNAMIC_REAL,
                    pred_length=self.prediction_length,
                    pad_value=0.0,
                )
            )
            dynamic_feat_fields.append(FieldName.PAST_FEAT_DYNAMIC_REAL)

        # we need to make sure that there is always some dynamic input
        # we will however disregard it in the hybrid forward
        if len(dynamic_feat_fields) == 0:
            chain.append(
                AddConstFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_CONST,
                    pred_length=self.prediction_length,
                    const=0.0,  # For consistency in case with no dynamic features
                    dtype=self.dtype,
                ),
            )
            dynamic_feat_fields.append(FieldName.FEAT_CONST)

        # now we map all the dynamic input of length context_length + prediction_length onto FieldName.FEAT_DYNAMIC
        # we exclude past_feat_dynamic_real since its length is only context_length
        if len(dynamic_feat_fields) > 1:
            chain.append(
                VstackFeatures(
                    output_field=FieldName.FEAT_DYNAMIC_REAL,
                    input_fields=dynamic_feat_fields,
                )
            )
        elif len(dynamic_feat_fields) == 1:
            chain.append(
                RenameFields(
                    {dynamic_feat_fields[0]: FieldName.FEAT_DYNAMIC_REAL}
                )
            )

        chain.append(
            InstanceSplitter(
                target_field=FieldName.TARGET,
                is_pad_field=FieldName.IS_PAD,
                start_field=FieldName.START,
                forecast_start_field=FieldName.FORECAST_START,
                train_sampler=ExpectedNumInstanceSampler(num_instances=1),
                past_length=self.context_length,
                future_length=self.prediction_length,
                time_series_fields=[
                    FieldName.OBSERVED_VALUES,
                    FieldName.FEAT_DYNAMIC_REAL,
                ],
            ),
        )

        dynamic_feat_fields = []

        padded_observed_field_name = f"past_{FieldName.OBSERVED_VALUES}"
        chain.append(
            FuturePadField(
                output_field=padded_observed_field_name,
                input_field=padded_observed_field_name,
                pred_length=self.prediction_length,
                pad_value=0.0,
            )
        )
        dynamic_feat_fields.append(padded_observed_field_name)

        padded_target_field_name = f"past_{FieldName.TARGET}"
        chain.append(
            FuturePadField(
                output_field=padded_target_field_name,
                input_field=padded_target_field_name,
                pred_length=self.prediction_length,
                pad_value=0.0,
            )
        )

        concatenated_feat_dynamic_field_name = (
            f"past_{FieldName.FEAT_DYNAMIC_REAL}"
        )
        chain.append(
            ConcatFields(
                input_fields=[
                    concatenated_feat_dynamic_field_name,
                    f"future_{FieldName.FEAT_DYNAMIC_REAL}",
                ],
                output_field=concatenated_feat_dynamic_field_name,
                axis=0,  # along time
            )
        )
        dynamic_feat_fields.append(concatenated_feat_dynamic_field_name)

        # STACK DYNAMIC VARIABLES INTO FieldName.FEAT_DYNAMIC

        chain.append(
            ConcatFields(
                input_fields=dynamic_feat_fields,
                output_field=f"past_{FieldName.FEAT_DYNAMIC_REAL}",
                axis=1,  # along features
            )
        )

        # CREATE DUMMY f"future_{FieldName.FEAT_DYNAMIC}"

        # override data with constants
        chain.append(
            # AddConstFeature(
            #     target_field=f"future_{FieldName.TARGET}",
            #     output_field=f"future_{FieldName.FEAT_DYNAMIC_REAL}",
            #     pred_length=self.prediction_length,
            #     const=0.0,  # For consistency in case with no dynamic features
            #     dtype=self.dtype,
            # )
            SetField(
                output_field=f"future_{FieldName.FEAT_DYNAMIC_REAL}",
                value=np.zeros((self.prediction_length, 1), dtype=self.dtype),
            )
        )

        # AT THIS POINT MAINLY
        #  f"past_{FieldName.FEAT_DYNAMIC}", # bunch of stuff concatenated and padded
        #  f"future_{FieldName.FEAT_DYNAMIC}" # dummy variable
        #  f"past_{FieldName.TARGET}" # padded
        #  f"future_{FieldName.TARGET}"
        #  f"future_{FieldName.OBSERVED_VALUES}"
        #  FieldName.FEAT_STATIC_CAT
        #  SHOULD be in the pipeline

        return transform.Chain(chain)

    def create_training_network(self) -> mx.gluon.HybridBlock:
        distribution = QuantileOutput(self.quantiles)

        # Actually pass through enc2dec is necessary for
        # CNN2QRForecaster to dump place holder future feat dynamic tensor for now
        enc2dec = PassThroughEnc2Dec()
        if self.decoder is None:
            decoder = OneShotDecoder(
                decoder_length=self.prediction_length,
                encoder_length=self.context_length,
                layer_sizes=self.decoder_mlp_layer,
                static_outputs_per_time_step=self.decoder_mlp_static_dim,
            )
        else:
            decoder = self.decoder

        training_network = Seq2SeqTrainingNetwork(
            embedder=self.embedder,
            scaler=self.scaler,
            encoder=self.encoder,
            enc2dec=enc2dec,
            decoder=decoder,
            quantile_output=distribution,
        )

        return training_network

    def create_predictor(
        self,
        transformation: transform.Transformation,
        trained_network: Seq2SeqTrainingNetwork,
    ) -> Predictor:
        # todo: this is specific to quantile output
        quantile_strs = [
            Quantile.from_float(quantile).name for quantile in self.quantiles
        ]

        prediction_network = Seq2SeqPredictionNetwork(
            embedder=trained_network.embedder,
            scaler=trained_network.scaler,
            encoder=trained_network.encoder,
            enc2dec=trained_network.enc2dec,
            decoder=trained_network.decoder,
            quantile_output=trained_network.quantile_output,
        )

        copy_parameters(trained_network, prediction_network)

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
            forecast_generator=QuantileForecastGenerator(quantile_strs),
        )


# TODO: fix mutable arguments
class MLP2QRForecaster(Seq2SeqEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int],
        embedding_dimension: List[int],
        encoder_mlp_layer: List[int],
        decoder_mlp_layer: List[int],
        decoder_mlp_static_dim: int,
        scaler: Scaler = NOPScaler(),
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        trainer: Trainer = Trainer(),
        num_parallel_samples: int = 100,
    ) -> None:
        encoder = MLPEncoder(layer_sizes=encoder_mlp_layer)
        super(MLP2QRForecaster, self).__init__(
            freq=freq,
            prediction_length=prediction_length,
            encoder=encoder,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_mlp_static_dim=decoder_mlp_static_dim,
            context_length=context_length,
            scaler=scaler,
            quantiles=quantiles,
            trainer=trainer,
            num_parallel_samples=num_parallel_samples,
        )


class RNN2QRForecaster(Seq2SeqEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        encoder_rnn_layer: int,
        encoder_rnn_num_hidden: int,
        cardinality: List[int] = None,
        embedding_dimension: List[int] = None,
        decoder_mlp_layer: List[int] = (30,),
        decoder_mlp_static_dim: int = 3,
        encoder_rnn_model: str = "lstm",
        encoder_rnn_bidirectional: bool = True,
        scaler: Scaler = NOPScaler(),
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        trainer: Trainer = Trainer(),
        num_parallel_samples: int = 100,
    ) -> None:
        encoder = RNNEncoder(
            mode=encoder_rnn_model,
            hidden_size=encoder_rnn_num_hidden,
            num_layers=encoder_rnn_layer,
            bidirectional=encoder_rnn_bidirectional,
            use_static_feat=True,
            use_dynamic_feat=True,
        )
        super(RNN2QRForecaster, self).__init__(
            freq=freq,
            prediction_length=prediction_length,
            encoder=encoder,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_mlp_static_dim=decoder_mlp_static_dim,
            context_length=context_length,
            scaler=scaler,
            quantiles=quantiles,
            trainer=trainer,
            num_parallel_samples=num_parallel_samples,
        )


class CNN2QRForecaster(Seq2SeqEstimator):
    @validated()
    def __init__(
        self,
        freq: str,
        prediction_length: int,
        cardinality: List[int] = None,
        embedding_dimension: List[int] = None,
        decoder_mlp_layer: List[int] = (30,),
        decoder_mlp_static_dim: int = 3,
        scaler: Scaler = NOPScaler(),
        context_length: Optional[int] = None,
        quantiles: Optional[List[float]] = None,
        trainer: Trainer = Trainer(),
        num_parallel_samples: int = 100,
        use_feat_static_cat: bool = False,
        add_time_feature: bool = True,
        add_age_feature: bool = False,
        use_past_feat_dynamic_real: bool = False,
        use_feat_dynamic_real: bool = False,
    ) -> None:
        encoder = HierarchicalCausalConv1DEncoder(
            dilation_seq=[1, 3, 9],
            kernel_size_seq=[7, 3, 3],
            channels_seq=[30, 30, 30],
            use_residual=True,
            use_dynamic_feat=True,
            use_static_feat=True,
        )

        context_length = (
            context_length
            if context_length is not None
            else 4 * prediction_length
        )
        decoder = OneShotDecoder(
            decoder_length=prediction_length,
            encoder_length=prediction_length + context_length,
            layer_sizes=decoder_mlp_layer,
            static_outputs_per_time_step=decoder_mlp_static_dim,
        )

        super().__init__(
            freq=freq,
            prediction_length=prediction_length,
            context_length=context_length,
            encoder=encoder,
            decoder=decoder,
            cardinality=cardinality,
            embedding_dimension=embedding_dimension,
            decoder_mlp_layer=decoder_mlp_layer,
            decoder_mlp_static_dim=decoder_mlp_static_dim,
            scaler=scaler,
            quantiles=quantiles,
            trainer=trainer,
            num_parallel_samples=num_parallel_samples,
            use_past_feat_dynamic_real=use_past_feat_dynamic_real,
            use_feat_dynamic_real=use_feat_dynamic_real,
            use_feat_static_cat=use_feat_static_cat,
            add_time_feature=add_time_feature,
            add_age_feature=add_age_feature,
        )

    @classmethod
    def derive_auto_fields(cls, train_iter):
        stats = calculate_dataset_statistics(train_iter)

        return {
            "use_past_feat_dynamic_real": stats.num_past_feat_dynamic_real > 0,
            "use_feat_dynamic_real": stats.num_feat_dynamic_real > 0,
            "use_feat_static_cat": bool(stats.feat_static_cat),
            "cardinality": [len(cats) for cats in stats.feat_static_cat],
        }

    @classmethod
    def from_inputs(cls, train_iter, **params):
        logger = logging.getLogger(__name__)
        logger.info(
            f"gluonts[from_inputs]: User supplied params set to {params}"
        )
        # auto_params usually include `use_feat_dynamic_real`, `use_past_feat_dynamic_real`,
        # `use_feat_static_cat` and `cardinality`
        auto_params = cls.derive_auto_fields(train_iter)

        fields = [
            "use_feat_dynamic_real",
            "use_past_feat_dynamic_real",
            "use_feat_static_cat",
        ]
        # user defined arguments become implications
        for field in fields:
            if field in params.keys():
                is_params_field = (
                    params[field]
                    if type(params[field]) == bool
                    else strtobool(params[field])
                )
                if is_params_field and not auto_params[field]:
                    logger.warning(
                        f"gluonts[from_inputs]: {field} set to False since it is not present in the data."
                    )
                    params[field] = False
                    if field == "use_feat_static_cat":
                        params["cardinality"] = None
                elif (
                    field == "use_feat_static_cat"
                    and not is_params_field
                    and auto_params[field]
                ):
                    params["cardinality"] = None

        # user specified 'params' will take precedence:
        params = {**auto_params, **params}
        logger.info(
            f"gluonts[from_inputs]: use_past_feat_dynamic_real set to "
            f"'{params['use_past_feat_dynamic_real']}', use_feat_dynamic_real set to "
            f"'{params['use_feat_dynamic_real']}', and use_feat_static_cat set to "
            f"'{params['use_feat_static_cat']}' with cardinality of '{params['cardinality']}'"
        )
        return cls.from_hyperparameters(**params)
