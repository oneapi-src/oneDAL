# file: model.py
#===============================================================================
# Copyright 2019 Intel Corporation
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

from typing import (
    Any,
    Dict,
    List,
    Text,
    Union,
)

from collections import namedtuple

def has_fields(obj):
    return hasattr(obj, '__fields__')

def iter_fields(obj):
    for attr in dir(obj):
        is_public = not attr.startswith('_')
        is_callable = callable(getattr(obj, attr))
        if is_public and not is_callable:
            yield attr

def _iter_model_object(obj, is_root=True):
    if isinstance(obj, list):
        for v in obj:
            yield from _iter_model_object(v, is_root=False)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_model_object(v, is_root=False)
    elif has_fields(obj):
        if is_root:
            for field in iter_fields(obj):
                yield from _iter_model_object(getattr(obj, field),
                                              is_root=False)
        else:
            yield obj


class _ModelProperty(object):
    def __init__(self, var_name):
        self._var_name = var_name

    def __get__(self, instance, owner):
        if instance:
            return getattr(instance, self._var_name)
        return self

    def __set__(self, instance, value):
        if instance:
            return setattr(instance, self._var_name, value)
        return self


def model_object(cls):
    fields = tuple(iter_fields(cls))
    defaults = tuple(getattr(cls, f) for f in fields)

    for field, default in zip(fields, defaults):
        setattr(cls, field, _ModelProperty(f'_{field}'))

    # __init__
    s = f'def __init__(self, {", ".join(fields)}): \n'
    for field in fields:
        s += f'    self._{field} = {field}\n'

    # __repr__
    repr_str = ', '.join(f'{f}={{self._{f}}}' for f in fields)
    s += f'def __repr__(self):\n'
    s += f'    return f"{{type(self).__name__}}({repr_str})"'

    namespace = {'__name__': f'__model_{cls.__name__}'}
    exec(s, namespace)

    cls.__fields__ = fields
    cls.__init__ = namespace['__init__']
    cls.__init__.__defaults__ = defaults
    cls.__repr__ = namespace['__repr__']
    cls.iter = iter_fields
    return cls


@model_object
class Run(object):
    content: Text = None
    kind: Text = 'unknown'

@model_object
class Description(object):
    runs: List[Run] = []

@model_object
class Doc(object):
    invariants: List[Description] = []
    postconditions: List[Description] = []
    preconditions: List[Description] = []
    remarks: List[Description] = []
    description: Description = None

@model_object
class Location(object):
    file: Text = None
    line: int = -1
    bodyfile: Text = None
    bodyend: int = -1
    bodystart: int = -1

@model_object
class Parameter(object):
    name: Text = None
    type: Text = None
    default: Text = None
    description: Description = None

@model_object
class Typedef(object):
    doc: Doc = None
    name: Text = None
    type: Text = None
    location: Location = None
    declaration: Text = None
    template_declaration: Text = None
    template_parameters: List[Parameter] = []
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

@model_object
class EnumClassValue(object):
    doc: Doc = None
    name: Text = None

@model_object
class EnumClass(object):
    doc: Doc = None
    name: Text = None
    values: List[EnumClassValue] = []
    location: Location = None
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

@model_object
class Function(object):
    doc: Doc = None
    name: Text = None
    location: Location = None
    argstring: Text = None
    parameters: List[Parameter] = []
    declaration: Text = None
    return_type: Text = None
    template_parameters: List[Parameter] = []
    template_declaration: Text = None
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

@model_object
class Class(object):
    doc: Doc = None
    kind: Text = None
    name: Text = None
    location: Location = None
    functions: List[Function] = []
    static_functions: List[Function] = []
    declaration: Text = None
    template_parameters: List[Parameter] = []
    template_declaration: Text = None
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

@model_object
class ClassRef(object):
    name: Text = None
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

@model_object
class Namespace(object):
    doc: Doc = None
    name: Text = None
    typedefs: List[Typedef] = []
    location: Location = None
    functions: List[Function] = []
    class_refs: List[ClassRef] = []
    enum_classes: List[EnumClass] = []
    fully_qualified_name: Text = None
    parent_fully_qualified_name: Text = None

class Visitor(object):
    def enter(self, node): ...
    def leave(self, node): ...

def visit(node, visitor: Visitor):
    if visitor.enter(node):
        for child in _iter_model_object(node):
            visit(child, visitor)
        visitor.leave(node)
