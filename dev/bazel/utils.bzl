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

load("@bazel_skylib//lib:paths.bzl", _paths = "paths")
load("@bazel_skylib//lib:new_sets.bzl", _sets = "sets")
load("@bazel_skylib//lib:collections.bzl", _collections = "collections")

paths = _paths
sets = _sets
collections = _collections

def unique(iterable):
    """Remove duplicates from a list."""
    return collections.uniq(iterable)

def warn(msg):
    """Output warning."""
    yellow = "\033[1;33m"
    no_color = "\033[0m"
    print("\n%sWARNING:%s %s\n" % (yellow, no_color, msg))

def info(msg):
    """Output warning."""
    yellow = "\033[0;32m"
    no_color = "\033[0m"
    print("\n%sINFO:%s %s\n" % (yellow, no_color, msg))

def add_prefix(prefix, lst):
    return [ prefix + str(x) for x in lst ]

def substitude(string, substitutions={}):
    string_fmt = string
    for key, value in substitutions.items():
        string_fmt = string_fmt.replace(key, value)
    return string_fmt

def datestamp(repo_ctx):
    return repo_ctx.execute(["date", "+%Y%m%d"]).stdout.strip()

utils = struct(
    unique = unique,
    warn = warn,
    info = info,
    add_prefix = add_prefix,
    substitude = substitude,
    datestamp = datestamp,
)
