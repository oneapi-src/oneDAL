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

load("@onedal//dev/bazel:utils.bzl", "utils", "paths")

def _create_symlinks(repo_ctx, root, entries, substitutions={}):
    for entry in entries:
        entry_fmt = utils.substitude(entry, substitutions)
        src_entry_path = paths.join(root, entry_fmt)
        dst_entry_path = entry_fmt
        repo_ctx.symlink(src_entry_path, dst_entry_path)

def _download(repo_ctx):
    output = repo_ctx.path("archive")
    repo_ctx.download_and_extract(
        url = repo_ctx.attr.url,
        sha256 = repo_ctx.attr.sha256,
        output = output,
        stripPrefix = repo_ctx.attr.strip_prefix,
    )
    return str(output)

def _prebuilt_libs_repo_impl(repo_ctx):
    root = repo_ctx.os.environ.get(repo_ctx.attr.root_env_var)
    if not root:
        if repo_ctx.attr.url:
            root = _download(repo_ctx)
        elif repo_ctx.attr.fallback_root:
            root = repo_ctx.attr.fallback_root
        else:
            fail("Cannot locate {} dependency".format(repo_ctx.name))
    substitutions = {
        # TODO: Detect OS
        "%{os}": "lnx",
    }
    _create_symlinks(repo_ctx, root, repo_ctx.attr.includes, substitutions)
    _create_symlinks(repo_ctx, root, repo_ctx.attr.libs, substitutions)
    repo_ctx.template(
        "BUILD",
        repo_ctx.attr.build_template,
        substitutions = substitutions,
    )

def prebuilt_libs_repo_rule(includes, libs, build_template,
                            root_env_var="", fallback_root="",
                            url="", sha256="", strip_prefix=""):
    return repository_rule(
        implementation = _prebuilt_libs_repo_impl,
        environ = [
            root_env_var,
        ],
        local = True,
        configure = True,
        attrs = {
            "root_env_var": attr.string(default=root_env_var),
            "fallback_root": attr.string(default=fallback_root),
            "url": attr.string(default=url),
            "sha256": attr.string(default=sha256),
            "strip_prefix": attr.string(default=strip_prefix),
            "includes": attr.string_list(default=includes),
            "libs": attr.string_list(default=libs),
            "build_template": attr.label(allow_files=True,
                                         default=Label(build_template)),
        }
    )

repos = struct(
    prebuilt_libs_repo_rule = prebuilt_libs_repo_rule,
)
