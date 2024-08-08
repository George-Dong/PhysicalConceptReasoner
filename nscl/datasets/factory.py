#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : factory.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 11/10/2018
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

from jacinle.logging import get_logger
from jacinle.utils.registry import RegistryGroup, CallbackRegistry

logger = get_logger(__file__)

__all__ = [
    'initialize_dataset',
    'get_available_datasets', 'get_dataset_builder',
    'get_available_symbolic_datasets', 'get_symbolic_dataset_builder',
    'get_available_concept_retrieval_datasets', 'get_concept_retrieval_dataset_builder',
    'get_available_concept_quantization_datasets', 'get_concept_quantization_dataset_builder',
    'register_dataset'
]

"""
Function template:

    >>> def build_xxx_dataset(args, configs, image_root, scenes_json, questions_json):
    >>>     pass
    >>> def build_concept_retrieval_xxx_dataset(args, configs, program, image_root, scenes_json):
    >>>     pass
    >>> def build_concept_quantization_xxx_dataset(args, configs, image_root, scenes_json):
    >>>     pass
    >>> def build_symbolic_minecraft_dataset(args):
    >>>     pass
"""


class DatasetRegistry(RegistryGroup):
    __base_class__ = CallbackRegistry


#import pdb
#pdb.set_trace()
dataset_registry = DatasetRegistry()


def initialize_dataset(dataset, version='v1'):
    from nscl.datasets.definition import set_global_definition
    if dataset=='clevrer' and version =='v1':
        from  clevrer.definition_clevrer import CLEVRERDefinition
        def_class = CLEVRERDefinition
    elif dataset=='clevrer' and (version =='v2' or version == 'v3' or version =='v4' or version=='v2_1'):
        from  clevrer.definition_clevrer_v2 import CLEVRERDefinitionV2
        def_class = CLEVRERDefinitionV2
    elif dataset=='billiards':
        from  clevrer.definition_billiards import BilliardDefinition
        def_class = BilliardDefinition 
    elif dataset=='blocks':
        from  clevrer.definition_blocks import BlockDefinition
        def_class = BlockDefinition 
    elif dataset=='comphy':
        from  clevrer.definition_comphy_v2 import ComPhyDefinition
        def_class = ComPhyDefinition
    elif dataset=='magnet':
        from  clevrer.definition_comphy_v2 import ComPhyDefinition
        def_class = ComPhyDefinition
    else:
        def_class = dataset_registry.lookup('definition', dataset, fallback=False)
    if def_class is None:
        raise ValueError('Unknown dataset: {}.'.format(dataset))
    set_global_definition(def_class())


def get_available_datasets():
    #import pdb
    #pdb.set_trace()
    return dataset_registry['dataset'].keys()


def get_dataset_builder(dataset):
    builder = dataset_registry.lookup('dataset', dataset, fallback=False)
    if builder is None:
        raise ValueError('Unknown dataset: {}.'.format(dataset))
    return builder


def get_available_symbolic_datasets():
    return dataset_registry['symbolic_dataset'].keys()


def get_symbolic_dataset_builder(dataset):
    builder = dataset_registry.lookup('symbolic_dataset', dataset, fallback=False)
    if builder is None:
        raise ValueError('Unknown dataset: {}.'.format(dataset))
    return builder


def get_available_concept_retrieval_datasets():
    return dataset_registry['concept_retrieval_dataset'].keys()


def get_concept_retrieval_dataset_builder(dataset):
    builder = dataset_registry.lookup('concept_retrieval_dataset', dataset, fallback=False)
    if builder is None:
        raise ValueError('Unknown dataset: {}.'.format(dataset))
    return builder


def get_available_concept_quantization_datasets():
    return dataset_registry['concept_quantization_dataset'].keys()


def get_concept_quantization_dataset_builder(dataset):
    builder = dataset_registry.lookup('concept_quantization_dataset', dataset, fallback=False)
    if builder is None:
        raise ValueError('Unknown dataset: {}.'.format(dataset))
    return builder


def register_dataset(name, def_class,
        builder=None, symbolic_builder=None,
        concept_retrieval_builder=None, concept_quantization_builder=None):

    dataset_registry.register('definition', name, def_class)
    for typename, builder_func in zip(
        ['dataset', 'symbolic_dataset', 'concept_retrieval_dataset', 'concept_quantization_dataset'],
        [builder, symbolic_builder, concept_retrieval_builder, concept_quantization_builder]
    ):
        if builder is not None:
            dataset_registry.register(typename, name, builder_func)

