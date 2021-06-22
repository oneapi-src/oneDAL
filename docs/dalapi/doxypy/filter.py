# file: filter.py
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


import re
from typing import (
    List,
    Tuple,
    Union,
    Dict,
    Text,
    NewType
)
from .parser import UndocumentedWarn

IgnoreConfigT = NewType("IgnoreConfigT", Tuple[Text, Union[Text, Dict[Text, Text]]])

class UndocumentedWarnFilter:

    PROP_MAPPING = {
        'Compound': {
            'identifier': 'member'
        },
        'Param': {
            'function': 'parent'
        }
    }

    MAIN_PROP = {
        'Param': 'parent',
        'Compound': 'member',
        'General': 'parent'
    }

    def __init__(self, ignore_configs: List[IgnoreConfigT]):
        self._ignore_filters: Dict[Text, List[Dict[Text, re.Pattern]]] = {
            'Compound': [],
            'Param': [],
            'General': []
        }
        self._init_ignore_filters(ignore_configs)

    def ignored(self, warn: UndocumentedWarn) -> bool:
        for filters in self._ignore_filters[warn.type_name]:
            for prop, pattern in filters.items():
                if not pattern.fullmatch(getattr(warn, prop)):
                    break
            else:
                return True
        return False

    def _init_ignore_filters(self, ignore_configs: List[IgnoreConfigT]):
        for type_name, config in ignore_configs:
            if type_name not in self._ignore_filters:
                raise ValueError(f"invalid ignore config type_name: {type_name}")
            if isinstance(config, str):
                prop = self.MAIN_PROP[type_name]
                self._ignore_filters[type_name].append({prop: re.compile(config)})
            elif isinstance(config, dict):
                self._ignore_filters[type_name].append({
                    self._map_prop(type_name, prop): re.compile(regex)
                        for prop, regex in config.items()
                })
            else:
                raise ValueError(f"unknown config: {config}, type: {type(config)}")

    @classmethod
    def _map_prop(cls, type_name, prop):
        return cls.PROP_MAPPING.get(type_name, {}).get(prop, prop)
