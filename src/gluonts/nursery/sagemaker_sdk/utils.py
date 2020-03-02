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
import re
import tarfile
from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path
import io

# First-party imports
from gluonts.model.predictor import Predictor

# Third-party imports
import s3fs
import pandas as pd


def make_metrics(metrics_names):
    avg_epoch_loss_metric = {
        "Name": "training_loss",
        "Regex": r"'epoch_loss'=(\S+)",
    }
    final_loss_metric = {"Name": "final_loss", "Regex": r"Final loss: (\S+)"}
    other_metrics = [
        {
            "Name": metric,
            "Regex": rf"gluonts\[metric-{re.escape(metric)}\]: (\S+)",
        }
        for metric in metrics_names
    ]

    return [avg_epoch_loss_metric, final_loss_metric] + other_metrics


def make_job_name(base_job_name):
    now = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")
    return f"{base_job_name}-{now}"


def get_s3_web_link(experiment_dir: str, job_name: str, region_name: str):
    return (
        f"https://s3.console.aws.amazon.com/s3/buckets/{experiment_dir[5:]}/{job_name}/?region={region_name}&tab"
        f"=overview "
    )


def get_training_job_web_link(region_name: str, job_name: str):
    return f"https://{region_name}.console.aws.amazon.com/sagemaker/home?region={region_name}#/jobs/{job_name}"


def retrieve_file_from_output(
    s3_output_path: str,
    job_name: str,
    file_location: str,
    file_type: str = "csv",
    **kwargs,
) -> io.BufferedReader:
    """
    Retrieve any file from the output.
    Currently only 'csv' 'file_type' supported.

    Parameters
    ----------
    s3_output_path
        The output_path for your experiment.
    job_name
        The s3 path-stype URL to 'output.tar.gz'.
    file_location
        The relative path to the file within the 'output' directory.
    file_type
        Currently only supports csv
    kwargs
        Arguments passed to 's3fs.S3FileSystem'.

    Returns
    -------
    io.BufferedReader

    """
    assert s3_output_path.startswith(
        "s3://"
    ), "The location must be a valid s3 file path starting with 's3://'."
    assert file_type == "csv"

    output_tar_gz_location = f"{s3_output_path}/{job_name}/output.tar.gz"

    with s3fs.S3FileSystem(**kwargs).open(
        output_tar_gz_location, "rb"
    ) as stream:
        with tarfile.open(fileobj=stream, mode="r:gz") as archive:
            if file_type == "csv":
                file = pd.read_csv(archive.extractfile(file_location))
            else:
                raise AssertionError(
                    "No other file types than 'csv' currently supported."
                )
    return file


def retrieve_model(s3_output_path: str, job_name: str, **kwargs) -> Predictor:
    """
    Retrieve the Predictor that was serialized to the 'model' directory during training with GluonTSFramework.

    Parameters
    ----------
    s3_output_path
        The output_path for your experiment.
    job_name
        The s3 path-stype URL to 'model.tar.gz'.
    kwargs
        Arguments passed to 's3fs.S3FileSystem'.

    Returns
    -------
    Predictor

    """
    assert s3_output_path.startswith(
        "s3://"
    ), "The location must be a valid s3 file path starting with 's3://'."

    model_tar_gz_location = f"{s3_output_path}/{job_name}/model.tar.gz"

    with s3fs.S3FileSystem(**kwargs).open(
        model_tar_gz_location, "rb"
    ) as stream:
        with tarfile.open(mode="r:gz", fileobj=stream) as archive:
            with TemporaryDirectory() as temp_dir:
                archive.extractall(temp_dir)
                predictor = Predictor.deserialize(Path(temp_dir))

    return predictor
