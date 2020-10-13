# SPDX-FileCopyrightText: 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
import re
from typing import (
    List,
    Text,
    Union,
)
from . import model
from . import utils

class DeclarationError(Exception):
    pass


# _getter_re = re.compile(r' get_\w+ *\(')
# _setter_re = re.compile(r' set_\w+ *\(')
# def _group_accessors(self, listing):
#     getter_re = FileListingReader._getter_re
#     setter_re = FileListingReader._setter_re
#     getter_match = False
#     setter_match = False
#     line_stack = []
#     for line in listing:
#         if line:
#             local_getter_match = getter_re.search(line)
#             local_setter_match = setter_re.search(line)
#             if (getter_match and local_getter_match or
#                 setter_match and local_setter_match):
#                 if line_stack[-1] == '':
#                     line_stack.pop()
#             getter_match = local_getter_match
#             setter_match = local_setter_match
#         line_stack.append(line)
#     return line_stack


class _ListingEntry(object):
    def __init__(self, base_dir, model_object):
        self._base_dir = base_dir
        self._filename = model_object.location.file
        self._model_object = model_object
        self._content = None

    @utils.return_list
    def read(self, remove_empty_lines: bool):
        if not self._content:
            filename = os.path.join(self._base_dir, self._filename)
            with open(filename, 'r') as f:
                self._content = self._read(f.readlines())
        if remove_empty_lines:
            for line in self._content:
                if line.strip():
                    yield line
        else:
            yield from self._content

    @utils.return_list
    def _read(self, lines):
        start_index = self._find_start_index(lines)
        if start_index is None:
            raise DeclarationError(
                f'Cannot find the beginning of declaration for the'
                f'{self._model_object.fully_qualified_name} in {self._filename}'
            )
        end_index = self._find_end_index(lines, start_index)
        if end_index is None:
            raise DeclarationError(
                f'Cannot find the end of declaration for the '
                f'{self._model_object.fully_qualified_name} in {self._filename}'
            )
        for i in range(start_index, end_index + 1):
            line = lines[i].strip()
            if not line.startswith('/'):
                yield lines[i].rstrip()

    def _find_start_index(self, lines):
        assert self._model_object.location.line > 0
        line_index = self._model_object.location.line - 1
        if line_index == 0:
            return 0
        # Note: Handles only the case if there is a blank
        # line or comment before the declaration
        for i in range(line_index, -1, -1):
            line = lines[i].strip()
            if not line or line.startswith('/'):
                return i + 1

    def _find_end_index(self, lines, start_index):
        bodyend = self._model_object.location.bodyend
        if bodyend is not None and bodyend > start_index:
            return bodyend - 1
        if isinstance(self._model_object, model.Namespace):
            return self._find_namespace_end(lines, start_index)
        else:
            return self._find_line_index(lines, start_index, ';')

    def _find_namespace_end(self, lines, start_index):
        brace_counter = 0
        for i in range(start_index, len(lines)):
            brace_counter += lines[i].count('{')
            if brace_counter > 0:
                brace_counter -= lines[i].count('}')
                if brace_counter <= 0:
                    return i

    def _find_line_index(self, lines, start_index, trigger):
        for i in range(start_index, len(lines)):
            if trigger in lines[i]:
                return i


class ListingReader(object):
    def __init__(self, base_dir):
        self._base_dir = base_dir
        self._cache = {}

    def read(self, model_object, remove_empty_lines=False) -> List[Text]:
        fqn = model_object.fully_qualified_name
        if fqn not in self._cache:
            self._cache[fqn] = _ListingEntry(self._base_dir, model_object)
        return self._cache[fqn].read(remove_empty_lines)
