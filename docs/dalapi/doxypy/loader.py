# file: loader.py
#===============================================================================
# Copyright 2019-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

from typing import List
from . import model
from .parser import Parser
from .builder import build, ModelBuilder

class TransformerPass(object):
    def enter(self, node): ...
    def transform(self, node): ...

class NameTransformer(object):
    def transform(self, fully_qualified_name): ...

class _NameTransformerPass(TransformerPass):
    def __init__(self, name_transformer: NameTransformer):
        assert name_transformer is not None
        self._name_transformer = name_transformer

    def enter(self, node):
        return True

    def transform(self, node):
        self._transform_name(node, 'fully_qualified_name')
        self._transform_name(node, 'parent_fully_qualified_name')

    def _transform_name(self, node, attribute):
        attr_value = getattr(node, attribute, None)
        if attr_value:
            transformed = self._name_transformer.transform(attr_value)
            setattr(node, attribute, transformed)

class Transformer(model.Visitor):
    default_passes = []

    def __init__(self, name_transformer: NameTransformer = None,
                       passes: List[TransformerPass] = []):
        assert passes is not None
        default_passes = Transformer.default_passes.copy()
        if name_transformer is not None:
            default_passes.append(_NameTransformerPass(name_transformer))
        self._passes = default_passes + passes
        self._name_transformer = name_transformer
        self._current_pass = None

    def transform(self, node):
        for transformer_pass in self._passes:
            self._current_pass = transformer_pass
            model.visit(node, self)

    def transform_name(self, fully_qualified_name):
        if self._name_transformer:
            return self._name_transformer.transform(fully_qualified_name)
        return fully_qualified_name

    def enter(self, node):
        return self._current_pass.enter(node)

    def leave(self, node):
        self._current_pass.transform(node)


class ModelLoader(object):
    def __init__(self, parser: Parser, transformer: Transformer):
        assert parser
        assert transformer
        self._parser = parser
        self._transformer = transformer
        self._cache = {}

    def load(self, refid):
        assert refid
        if refid == 'index':
            raise ValueError('Index cannot be loaded as a model')
        model = self._cache.get(refid, None)
        if not model:
            model = self._load(refid)
            self._cache[refid] = model
        return model

    def _load(self, refid):
        doxygen = self._parser.parse(refid)
        if len(doxygen.compounddef) != 1:
            raise ValueError(f'Cannot interpet Doxygen output, '
                             f'unexpected content of the {refid} file')
        node = build(ModelBuilder, doxygen.compounddef[0])
        self._transformer.transform(node)
        return node
