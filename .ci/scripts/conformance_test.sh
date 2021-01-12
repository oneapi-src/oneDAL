#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
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

while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --build-dir)
        BUILD_DIR="$2"
        ;;
        --conda-dir)
        CONDA_DIR="$2"
        ;;
        --oneapi-dir)
        ONEAPI_DIR="$2"
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
    shift
    shift
done

# testing
cd .ci/scripts/conformance-scripts/ || exit 1
export IDP_SKLEARN_VERBOSE=INFO
python run_tests.py ${PYTHON_VERSION} || exit 1
cd ../../..
