#!/bin/bash
#===============================================================================
# Copyright 2014-2021 Intel Corporation
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

# shellcheck shell=sh
# Get absolute pathname to script, when sourced from bash, zsh and ksh shells.
# see https://stackoverflow.com/a/29835459/2914328 for detailed "how it works"
#
# Attribution of this rreadlink() function goes to Michael Klement, aka
# "mklement0" on StackOverflow. Based on "edited Sep 16 '19" version of SO post.
# Based on https://github.com/mklement0/rreadlink/blob/master/bin/rreadlink#L125
# Above licensed under MIT license > https://spdx.org/licenses/MIT#licenseText
#     User profile: https://stackoverflow.com/users/45375/mklement0
#         LinkedIn: https://www.linkedin.com/in/mklement0/
#          SO post: https://stackoverflow.com/a/29835459/2914328
# SO License terms: https://stackoverflow.com/help/licensing
#     CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/
#
# Adapted and modified by Paul A. Fischer, Intel Corporation
#
# This POSIX-compliant shell function implements an equivalent to the GNU
# `readlink -e` command and is a reasonably robust solution that only fails
# in two rare edge cases:
#   * paths with embedded newlines (very rare)
#   * filenames containing literal string -> (also rare)

# Usage:
#   script_path=$(rreadlink "$vars_script_rel_path")
#   script_dir_path=$(dirname -- "$(rreadlink "$vars_script_rel_path")")
#
# Inputs:
#   script/relative/pathname/scriptname
#
# Outputs:
#   /script/absolute/pathname/scriptname

# executing function in a *subshell* to localize vars and effects on `cd`
rreadlink() (
  target=$1 fname="" targetDir="" CDPATH=
  { \unalias command; \unset -f command; } >/dev/null 2>&1 || :
  # shellcheck disable=SC2034
  [ -n "${ZSH_VERSION:-}" ] && options[POSIX_BUILTINS]=on
  while :; do
    [ -L "$target" ] || [ -e "$target" ] || { command printf '%s\n' ":: ERROR: rreadlink(): '$target' does not exist." >&2; return 1; }
    command cd "$(command dirname -- "$target")" >/dev/null 2>&1
    fname=$(command basename -- "$target")
    [ "$fname" = '/' ] && fname=''
    if [ -L "$fname" ] ; then
      target=$(command ls -l "$fname")
      target=${target#* -> } # delete everything left of first " -> " string
      continue
    fi
    break
  done
  targetDir=$(command pwd -P)
  if   [ "$fname" = '.' ] ;  then
    command printf '%s\n' "${targetDir%/}"
  elif [ "$fname" = '..' ] ; then
    command printf '%s\n' "$(command dirname -- "${targetDir}")"
  else
    command printf '%s\n' "${targetDir%/}/$fname"
  fi
)


# ############################################################################

# Check to insure that this script is being sourced, not executed.
# see https://stackoverflow.com/a/38128348/2914328
# see https://stackoverflow.com/a/28776166/2914328
# see https://stackoverflow.com/a/60783610/2914328
# see https://stackoverflow.com/a/2942183/2914328

# This script is designed to be POSIX compatible, there are a few lines in the
# code block below that contain elements that are specific to a shell. The
# shell-specific elements are needed to identify the sourcing shell.

vars_script_name="vars.sh"

vars_sourced=0 ;
vars_sourced_sh="$(ps -p "$$" -o  command= | awk '{print $1}')" ;
vars_sourced_nm="$(ps -p "$$" -o  command= | awk '{print $2}')" ;

# ${var:-} needed to pass "set -eu" checks
# see https://unix.stackexchange.com/a/381465/103967
# see https://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html#tag_18_06_02
if [ -n "${ZSH_VERSION:-}" ] ; then     # only executed in zsh
  vars_sourced=0 ; #echo "   ZSH version \"$ZSH_VERSION\""
  vars_sourced_sh="zsh" ;               # only meaningful if vars_sourced=1
  vars_sourced_nm="${(%):-%x}" ;        # ditto
  if [ -n "$ZSH_EVAL_CONTEXT" ] ; then  # only present in zsh 5.x and later
    case $ZSH_EVAL_CONTEXT in (*:file*) vars_sourced=1 ;; esac ;
  fi
elif [ -n "${KSH_VERSION:-}" ] ; then   # only executed in ksh or mksh or lksh
  vars_sourced=0 ; #echo "   KSH version \"$KSH_VERSION\""
  if [ "$(set | grep KSH_VERSION)" = "KSH_VERSION=.sh.version" ] ; then # ksh
    if [ "$(cd "$(dirname -- "$0")" && pwd -P)/$(basename -- "$0")" \
      != "$(cd "$(dirname -- "${.sh.file}")" && pwd -P)/$(basename -- "${.sh.file}")" ] ; then
      vars_sourced=1 ;
      vars_sourced_sh="ksh" ;           # only meaningful if sourced=1
      vars_sourced_nm="${.sh.file}" ;   # ditto
    fi
  else # mksh or lksh detected (also check for [lm]ksh renamed as ksh)
    vars_sourced_sh="$(basename -- "$0")"
    if [ "mksh" = "$vars_sourced_sh" ] || [ "lksh" = "$vars_sourced_sh" ] || [ "ksh" = "$vars_sourced_sh" ] ; then
      vars_sourced=1 ;
      # force [lm]ksh to issue error msg; contains this script's rel/path/filename
      vars_sourced_nm="$( (echo "${.sh.file}") 2>&1 )" || : ;
      vars_sourced_nm="$(expr "$vars_sourced_nm" : '^.*ksh: \(.*\)\[[0-9]*\]:')" ;
    fi
  fi
elif [ -n "${BASH_VERSION:-}" ] ; then  # only executed in bash
  vars_sourced=0 ; #echo "   BASH version \"$BASH_VERSION\""
  vars_sourced_sh="bash" ;              # only meaningful if vars_sourced=1
  # shellcheck disable=2128
  vars_sourced_nm="$BASH_SOURCE" ;      # ditto
  (return 0 2>/dev/null) && vars_sourced=1 ;
# TODO: following needs further testing to work in dash
elif [ "${0:-}" = "dash" ] ; then
  vars_sourced=1 ;                      # see error messages below for outcome
  vars_sourced_sh="dash" ;              # see error messages below for outcome
  vars_sourced_nm="$vars_sourced_nm" ;
# Only reliable way to detect sh source is to know the name of this script.
# TODO: following needs further testing to work in sh
elif [ "$(basename -- "$0")" != "$vars_script_name" ] ; then
  vars_sourced=1 ;                  # see error messages below for outcome
  vars_sourced_sh="sh" ;            # see error messages below for outcome
  vars_sourced_nm="$vars_sourced_nm" ;
fi

if [ ${vars_sourced:-} -eq 0 ] ; then
  >&2 echo ":: ERROR: Incorrect usage: \"$vars_sourced_nm\" must be sourced." ;
  if [ "zsh" = "$vars_sourced_sh" ] ; then
    >&2 echo "   Sourcing in ZSH requires version 5+. You have version $ZSH_VERSION."
  fi
  # usage # if you want to add a usage() function, put a call to it here
  return 255 2>/dev/null || exit 255 ;
fi

if [ "" = "$vars_script_name" ] ; then
  >&2 echo ":: ERROR: Unable to proceed: no support for sourcing from '[dash|sh]' shell." ;
  >&2 echo "   This script must be sourced. Did you execute or source this script?" ;
  >&2 echo "   Can be caused by sourcing from inside a \"shebang-less\" script." ;
  >&2 echo "   Can also be caused by sourcing from ZSH version 4.x or older." ;
  return 1 2>/dev/null || exit 1
fi

__daal_tmp_dir="<INSTALLDIR>"
__daal_tmp_dir=$__daal_tmp_dir/dal
if [ ! -d $__daal_tmp_dir ]; then
    __daal_tmp_dir=$(dirname -- "$(get_script_path "$vars_script_name")")
fi

export DAL_MAJOR_BINARY=__DAL_MAJOR_BINARY__
export DAL_MINOR_BINARY=__DAL_MINOR_BINARY__
export DALROOT=$__daal_tmp_dir
export DAALROOT=$__daal_tmp_dir
export CPATH=$__daal_tmp_dir/include${CPATH+:${CPATH}}
export LIBRARY_PATH=$__daal_tmp_dir/lib${LIBRARY_PATH+:${LIBRARY_PATH}}
export DYLD_LIBRARY_PATH=$__daal_tmp_dir/lib${DYLD_LIBRARY_PATH+:${DYLD_LIBRARY_PATH}}
export CLASSPATH=$__daal_tmp_dir/lib/onedal.jar${CLASSPATH+:${CLASSPATH}}
