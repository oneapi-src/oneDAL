# file: __init__.py
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

import os
from typing import (
    Any,
    Dict,
    List,
    Text,
    Union,
)
from .index import parse as parse_index
from .compound import parse as parse_compound
from .warning_log import parse as parse_warning


class Parser(object):
    def __init__(self, xml_dir: Text):
        self._dir = xml_dir

    def parse(self, refid):
        xml_filename = self._resolve_path(refid)
        parse = parse_index if refid == 'index' else parse_compound
        return parse(xml_filename, silence=True)

    def parse_warning(self):
        warning_log_path = os.path.join(os.path.dirname(self._dir), 'warning.log')
        if not os.path.exists(warning_log_path):
            return {}
        return parse_warning(warning_log_path)

    def _resolve_path(self, refid):
        xml_name = os.path.join(self._dir, refid)
        return f'{xml_name}.xml'
