# SPDX-FileCopyrightText: 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

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

class Parser(object):
    def __init__(self, xml_dir: Text):
        self._dir = xml_dir

    def parse(self, refid):
        xml_filename = self._resolve_path(refid)
        parse = parse_index if refid == 'index' else parse_compound
        return parse(xml_filename, silence=True)

    def _resolve_path(self, refid):
        xml_name = os.path.join(self._dir, refid)
        return f'{xml_name}.xml'
