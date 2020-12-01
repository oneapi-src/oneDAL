# file: utils.py
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

def split_compound_name(compoundname):
    try:
        namespace, name = compoundname.rsplit('::', 1)
    except ValueError:
        namespace, name = '', compoundname
    return namespace, name

def return_list(func):
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper

def return_dict(func):
    def wrapper(*args, **kwargs):
        return dict(func(*args, **kwargs))
    return wrapper
