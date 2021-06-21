#!/usr/bin/env python
# file: warning_log.py
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

# -*- coding: utf-8 -*-

from typing import List, Dict
import re


class RawWarn:
    def __init__(self, filepath, lineno, text):
        self.filepath = filepath
        self.lineno = lineno
        self.text = text
        self.subitems = []

    def add_subitem(self, subitem):
        self.subitems.append(subitem)

    def __str__(self):
        return f'{self.filepath}:{self.lineno}: {self.text} {self.subitems if self.subitems else ""}'


class ParsedWarn:
    def __init__(self, warn: RawWarn):
        self._filepath = warn.filepath
        self._lineno = warn.lineno
        self._raw = warn
        self._parse(warn)

    def __init_subclass__(cls, **kwargs):
        cls.__wname__ = cls.__name__[:-4]

    def __str__(self):
        return str(self._raw)

    def _parse(self, warn: RawWarn):
        raise NotImplementedError

    @staticmethod
    def is_instance(warn: RawWarn):
        raise NotImplementedError


class UnknownWarn(ParsedWarn):
    def _parse(self, warn: RawWarn):
        pass

    @staticmethod
    def is_instance(warn: RawWarn):
        return True


class UndocumentedWarn(ParsedWarn):

    TYPE_NORMAL = 0
    TYPE_PARAM = 1
    TYPE_COMPOUND = 2

    normal_regex = re.compile(r'^ warning: Member (?P<member>.+?) \((?P<member_type>\w+)\) of (?P<parent_type>\w+) (?P<parent>.+?) is not documented\.$')
    compound_regex = re.compile(r'^ warning: Compound (?P<member>.+?) is not documented\.$')
    param_regex = re.compile(r'^ warning: The following parameter of (?P<parent>.+?) is not documented:$')
    parameter_regex = re.compile(r"^parameter '(?P<parameter>[^']+')$")

    def _parse(self, warn: RawWarn):
        self.type = self.TYPE_NORMAL
        self.member = None
        self.member_type = None
        self.parent = None
        self.parent_type = None
        self.parameters = []

        if warn.text[-1] == ':':  # param
            self.type = self.TYPE_PARAM
            match = self.param_regex.match(warn.text)
            if not match:
                raise RuntimeError(f"Unknown UndocumentedWarn Format!: {warn}")
            self.parent = match.group('parent')
            self.parameter = [self.__extract_parameter(item, warn) for item in warn.subitems]
        elif warn.text[10:18] == 'Compound':
            self.type = self.TYPE_COMPOUND
            match = self.compound_regex.match(warn.text)
            if not match:
                raise RuntimeError(f"Unknown UndocumentedWarn Format!: {warn}")
            self.member = match.group('member')
        else:
            match = self.normal_regex.match(warn.text)
            if not match:
                raise RuntimeError(f"Unknown UndocumentedWarn Format!: {warn}")
            self.member = match.group('member')
            self.member_type = match.group('member_type')
            self.parent = match.group('parent')
            self.parent_type = match.group('parent_type')

    @classmethod
    def __extract_parameter(cls, item, warn):
        match = cls.parameter_regex.match(item)
        if not match:
            raise RuntimeError(f"Unknown UndocumentedWarn Paramter Format!: {item} of {warn}")
        return match.group('parameter')

    @staticmethod
    def is_instance(warn: RawWarn):
        return 'not documented' == warn.text[-15:-1]


WarnKlasses = [
    UndocumentedWarn,
    UnknownWarn
]


def parse_log(in_file) -> List[RawWarn]:
    warnings = []
    with open(in_file) as f:
        for line in f.readlines():
            items = line.strip().split(':', 2)
            if len(items) == 1:
                warnings[-1].add_subitem(line.strip())
            elif len(items) == 3:
                warnings.append(RawWarn(*items))
            else:
                raise RuntimeError(f"Unknown warning format! warning line: {line}")

    return warnings


def parse_warn(raw_warns: List[RawWarn]) -> Dict[str, List[ParsedWarn]]:
    parsed_warns = {warn_klass.__wname__: [] for warn_klass in WarnKlasses}
    for warn in raw_warns:
        for warn_klass in WarnKlasses:
            if not warn_klass.is_instance(warn):
                continue
            parsed_warns[warn_klass.__wname__].append(warn_klass(warn))
            break

    return parsed_warns


def parse(in_file):
    return parse_warn(parse_log(in_file))
