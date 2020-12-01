# file: cli.py
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
