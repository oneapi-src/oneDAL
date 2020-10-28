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

# ############################################################################
# Get absolute path to script, when sourced from bash, zsh and ksh shells.
# Uses `readlink` to remove links and `pwd -P` to turn into an absolute path.
# Derived from similar function used by VTune and Advisor.
# Converted into a POSIX-compliant function.

# Usage:
#   script_dir=$(get_script_path "$script_rel_path")
#
# Inputs:
#   script/relative/pathname/scriptname
#
# Outputs:
#   /script/absolute/pathname
# executing function in a *subshell* to localize vars and effects on `cd`
get_script_path() (
  script="$1"
  while [ -L "$script" ] ; do
    # combining next two lines fails in zsh shell
    script_dir=$(command dirname -- "$script")
    script_dir=$(cd "$script_dir" && command pwd -P)
    script="$(readlink "$script")"
    case $script in
      (/*) ;;
       (*) script="$script_dir/$script" ;;
    esac
  done
  # combining next two lines fails in zsh shell
  script_dir=$(command dirname -- "$script")
  script_dir=$(cd "$script_dir" && command pwd -P)
  echo "$script_dir"
)

# ############################################################################
# Even though this script is designed to be POSIX compatible, there are lines
# in the code block below that are _not_ POSIX compatible. This works within a
# POSIX compatible shell because they are single-pass interpreters. Each "if
# test" that checks for a non-POSIX shell (zsh, bash, etc.) will return a
# "false" condition in a POSIX shell and, thus, will skip the non-POSIX lines.
# This requires that the "if test" constructs _are_ POSIX compatible.

usage() {
  printf "%s\n"   "ERROR: This script must be sourced."
  printf "%s\n"   "Usage: source $1"
  return 2 2>/dev/null || exit 2
}

if [ -n "$ZSH_VERSION" ] ; then
  # shellcheck disable=2039,2015  # following only executed in zsh
  [[ $ZSH_EVAL_CONTEXT =~ :file$ ]] && vars_script_name="${(%):-%x}" || usage "${(%):-%x}"
elif [ -n "$KSH_VERSION" ] ; then
  # shellcheck disable=2039,2015  # following only executed in ksh
  [[ $(cd "$(dirname -- "$0")" && printf '%s' "${PWD%/}/")$(basename -- "$0") != \
  "${.sh.file}" ]] && vars_script_name="${.sh.file}" || usage "$0"
elif [ -n "$BASH_VERSION" ] ; then
  # shellcheck disable=2039,2015  # following only executed in bash
  (return 0 2>/dev/null) && vars_script_name="${BASH_SOURCE[0]}" || usage "${BASH_SOURCE[0]}"
else
  case ${0##*/} in (sh|dash) vars_script_name="" ;; esac
fi

if [ "" = "$vars_script_name" ] ; then
  >&2 echo ":: ERROR: Unable to proceed: no support for sourcing from '[dash|sh]' shell." ;
  >&2 echo "   Can be caused by sourcing from inside a \"shebang-less\" script." ;
  return 1
fi

__daal_tmp_dir="<INSTALLDIR>"
__daal_tmp_dir=$__daal_tmp_dir/dal
if [ ! -d $__daal_tmp_dir ]; then
    __daal_tmp_dir=$(dirname -- "$(get_script_path "$vars_script_name")")
fi

export DAL_MAJOR_BINARY=1
export DAL_MINOR_BINARY=0
export DALROOT=$__daal_tmp_dir
export DAALROOT=$__daal_tmp_dir
export CPATH=$__daal_tmp_dir/include${CPATH+:${CPATH}}
export LIBRARY_PATH=$__daal_tmp_dir/lib/intel64${LIBRARY_PATH+:${LIBRARY_PATH}}
export LD_LIBRARY_PATH=$__daal_tmp_dir/lib/intel64${LD_LIBRARY_PATH+:${LD_LIBRARY_PATH}}
export CLASSPATH=$__daal_tmp_dir/lib/onedal.jar${CLASSPATH+:${CLASSPATH}}
