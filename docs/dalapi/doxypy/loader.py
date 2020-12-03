# file: loader.py
#===============================================================================
# Copyright 2019-2020 Intel Corporation
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

class Transformer(model.Visitor):
    default_passes = []

    def __init__(self, passes: List[TransformerPass] = []):
        assert passes is not None
        self._passes = Transformer.default_passes + passes
        self._current_pass = None

    def transform(self, node):
        for transformer_pass in self._passes:
            self._current_pass = transformer_pass
            model.visit(node, self)

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
