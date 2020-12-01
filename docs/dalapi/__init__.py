# SPDX-FileCopyrightText: 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

__version__ = '0.0.1'

def setup(app):
    from . import extension
    extension.setup(app)
    return {
        'version': __version__,
        'parallel_read_safe': True,
        'parallel_write_safe': True
    }
