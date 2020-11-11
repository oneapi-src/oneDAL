# SPDX-FileCopyrightText: 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

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
