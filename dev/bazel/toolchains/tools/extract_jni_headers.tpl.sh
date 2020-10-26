#!/bin/bash
#===============================================================================
# Copyright 2020 Intel Corporation
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

# Input format
# $1: jar archive that contains all JNI headers
# $2: output directory
# $3...$n: expected JNI headers list

jar_archive=$1
output_dir=$2
shift 2

# Extact JNI headers
%{jar_path} xf ${jar_archive}

# If some expected headers are missing
# in the archive, we will create dummy files
error_message="#error \"There are no native functions\
 declared in the corresponding java file\""
for jni_header in $*
do
    if [ ! -f "${jni_header}" ]; then
        echo ${error_message} > "${jni_header}"
    fi
done

# Move all files to the output directory
mkdir -p ${output_dir}
mv $* ${output_dir}
