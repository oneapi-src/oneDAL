# SPDX-FileCopyrightText: 2019-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT

from .index import index, to_yaml

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser('doxypy: The Doxygen parser for Python')
    args.add_argument('dir', type=str,
                      help='Path to the directory with Doxygen XML output')
    args.add_argument('--compact', action='store_true', default=False,
                      help='Does not include empty fields into output')
    # TODO
    # args.add_argument('--only-functions', action='store_true', default=False,
    #                   help='If provided, displays only functions')
    # args.add_argument('--filter', type=str, default=None,
    #                   help='Enables parsing only for the matched names')
    config = args.parse_args()

    idx = index(config.dir)
    print(to_yaml(idx, discard_empty=config.compact, indent=2))
