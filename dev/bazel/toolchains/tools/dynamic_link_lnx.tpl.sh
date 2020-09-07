#!/bin/bash
#===============================================================================
# Copyright 2014-2020 Intel Corporation
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

args_patched=()
for arg in "$@"
do
    if [[ $arg == *".def" ]]; then
        def_file=${arg//@}
    else
        args_patched+=(${arg})
    fi
done

if [[ ${def_file} ]]; then
    export_symbols=$(grep -v -E '^(EXPORTS|;|$)' ${def_file} | sed -e 's/^/-u /')
    echo ${export_symbols} > ${def_file}_patched
    %{cc_path} "${args_patched[@]}" "@${def_file}_patched"
else
    %{cc_path} "$@"
fi
