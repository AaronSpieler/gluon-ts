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

from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np

from gluonts.core.component import validated, DType
from gluonts.dataset.common import DataEntry
from gluonts.transform import target_transformation_length

from ._base import MapTransformation, SimpleTransformation


class RenameFields(SimpleTransformation):
    """
    Rename fields using a mapping, if source field present.

    Parameters
    ----------
    mapping
        Name mapping `input_name -> output_name`
    """

    @validated()
    def __init__(self, mapping: Dict[str, str]) -> None:
        self.mapping = mapping
        values_count = Counter(mapping.values())
        for new_key, count in values_count.items():
            assert count == 1, f"Mapped key {new_key} occurs multiple time"

    def transform(self, data: DataEntry):
        for key, new_key in self.mapping.items():
            if key in data:
                # no implicit overriding
                assert new_key not in data
                data[new_key] = data[key]
                del data[key]
        return data


class RemoveFields(SimpleTransformation):
    """"
    Remove field names if present.

    Parameters
    ----------
    field_names
        List of names of the fields that will be removed
    """

    @validated()
    def __init__(self, field_names: List[str]) -> None:
        self.field_names = field_names

    def transform(self, data: DataEntry) -> DataEntry:
        for k in self.field_names:
            data.pop(k, None)
        return data


class SetField(SimpleTransformation):
    """
    Sets a field in the dictionary with the given value.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    @validated()
    def __init__(self, output_field: str, value: Any) -> None:
        self.output_field = output_field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        data[self.output_field] = self.value
        return data


class SetFieldIfNotPresent(SimpleTransformation):
    """Sets a field in the dictionary with the given value, in case it does not
    exist already.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    value
        Value to be set
    """

    @validated()
    def __init__(self, field: str, value: Any) -> None:
        self.output_field = field
        self.value = value

    def transform(self, data: DataEntry) -> DataEntry:
        if self.output_field not in data.keys():
            data[self.output_field] = self.value
        return data


class SelectFields(MapTransformation):
    """
    Only keep the listed fields

    Parameters
    ----------
    input_fields
        List of fields to keep.
    """

    @validated()
    def __init__(self, input_fields: List[str]) -> None:
        self.input_fields = input_fields

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        return {f: data[f] for f in self.input_fields}


class FuturePadField(MapTransformation):
    """
    Pads the feature by appending constant values.

    Fields with value ``None`` are ignored.

    Parameters
    ----------
    output_field
        Field name to use for the output
    target_field
        Field with target values (array) of time series
    input_field
        Fields name of feature to pad
    pred_length
        Prediction length
    dtype
        DType to create padding with
    pad_value
        By default 0.0
    """

    @validated()
    def __init__(
        self,
        output_field: str,
        input_field: str,
        pred_length: int,
        target_field: Optional[str] = None,
        dtype: DType = np.float32,
        pad_value: float = 0.0,
    ) -> None:
        self.output_field = output_field
        self.target_field = target_field
        self.input_field = input_field
        self.pred_length = pred_length
        self.feature_name = output_field
        self.dtype = dtype
        self.pad_value = pad_value

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        if self.target_field is not None:
            target_length = target_transformation_length(
                data[self.target_field], self.pred_length, is_train=is_train
            )
        else:
            target_length = len(data[self.input_field]) + self.pred_length

        # pad data up to target length
        data[self.output_field] = np.pad(
            data[self.input_field],
            (
                0,
                target_length - len(data[self.input_field]),
            ),  # only pad at right end
            "constant",
            constant_values=(0, self.pad_value),  # pad with zeros
        ).astype(dtype=self.dtype)
        # remove slack
        if self.input_field != self.output_field:
            del data[self.input_field]

        return data


class ConcatFields(SimpleTransformation):
    """Concatenate arrays along the desired axis,
    and expands them to 2D if necessary.

    Parameters
    ----------
    output_field
        Name of the field that will be set
    input_fields
        Name of the two fields to concatenate
    axis
        Value to be set
    """

    @validated()
    def __init__(
        self, output_field: str, input_fields: List[str], axis: int = 0
    ) -> None:
        self.output_field = output_field
        self.input_fields = input_fields
        self.axis = axis

    def transform(self, data: DataEntry) -> DataEntry:
        # expand dim if necessary
        data_to_concat = [data[field] for field in self.input_fields]
        data_to_concat_correct_dim = []
        for entry in data_to_concat:
            if len(entry.shape) != 2:
                data_to_concat_correct_dim.append(
                    np.expand_dims(entry, axis=1)
                )
            else:
                data_to_concat_correct_dim.append(entry)

        # concat data
        data[self.output_field] = np.concatenate(
            data_to_concat_correct_dim, axis=self.axis
        )

        # remove slack
        for field in self.input_fields:
            if field != self.output_field:
                del data[field]

        return data
