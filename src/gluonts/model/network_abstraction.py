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

### IDEA ###

"""
- have an abstraction for the framework dependent network
    - i.e. mxnet.gluon.Block, mx.gluon.SymbolBlock or pytorch.nn.Module
- should cover how to serialize and deserialize it
- should cover how to run inference on it
    - i.e. batch_input -> network_abstraction -> batch_output
- and given an appropriate subclass of Trainer, how to train it
    - i.e. abstract_trained_network = network_abstraction.train(Appropriate_Trainer, **train_args)
    - actually we already have TrainOutput, but for now it depends on the framework
"""
