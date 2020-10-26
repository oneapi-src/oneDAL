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

load("@onedal//dev/bazel:utils.bzl", "paths")

_JAVA_HOME_ERROR_MESSAGE = (
    "Make sure `JAVA_HOME` environment variable is" +
    "pointing to valid JDK installation."
)

def _detect_jdk_installation(repo_ctx):
    java_home = repo_ctx.os.environ["JAVA_HOME"]
    if java_home:
        return java_home
    javac_path = repo_ctx.which("javac")
    if javac_path:
        return str(javac_path.dirname.dirname.realpath)
    fail("Cannot local JDK installation for oneDAL. " +
         "Make sure `JAVA_HOME` environment variable " +
         "is set or `javac` is available in `PATH`")

def _validate_jdk_installation(repo_ctx, java_home):
    # Check if javac exists
    javac_path = repo_ctx.path(paths.join(java_home, "bin/javac"))
    if not javac_path.exists:
        fail("Cannot locate `javac`. " + _JAVA_HOME_ERROR_MESSAGE)
    # Check if lib directory exists
    lib_path = repo_ctx.path(paths.join(java_home, "lib"))
    if not lib_path.exists:
        fail("Cannot locate `lib` directory. " + _JAVA_HOME_ERROR_MESSAGE)
    # Check if jni.h exists
    jnih_path = repo_ctx.path(paths.join(java_home, "include/jni.h"))
    if not jnih_path.exists:
        fail("Cannot locate `jni.h` directory. " + _JAVA_HOME_ERROR_MESSAGE)
    # TODO: Check JDK version

def _get_file_path(repo_ctx, java_home, relative_tool_path):
    tool_path = repo_ctx.path(paths.join(java_home, relative_tool_path))
    if not tool_path.exists:
        fail("Cannot locate `{}`. ".format(tool_path.basename) + _JAVA_HOME_ERROR_MESSAGE)
    return str(tool_path.realpath)

def _create_extract_jni_headers_tool(repo_ctx, java_home):
    jar_path = _get_file_path(repo_ctx, java_home, 'bin/jar')
    extract_jni_header = "extract_jni_headers.sh"
    repo_ctx.template(
        extract_jni_header,
        Label("@onedal//dev/bazel/toolchains/tools:extract_jni_headers.tpl.sh"),
        {
            "%{jar_path}": jar_path,
        }
    )
    return str(repo_ctx.path(extract_jni_header))

def _symlink_jni_includes(repo_ctx, java_home):
    jni_include_path = _get_file_path(repo_ctx, java_home, 'include')
    repo_ctx.symlink(jni_include_path, "include")

def _onedal_jdk_toolchain_impl(repo_ctx):
    java_home = _detect_jdk_installation(repo_ctx)
    _validate_jdk_installation(repo_ctx, java_home)
    _symlink_jni_includes(repo_ctx, java_home)
    repo_ctx.template(
        "BUILD",
        Label("@onedal//dev/bazel/toolchains/jdk:toolchain.tpl.BUILD"),
        {
            "%{java_home}": java_home,
            "%{extract_jni_headers}": _create_extract_jni_headers_tool(repo_ctx, java_home),
        }
    )

onedal_jdk_toolchain = repository_rule(
    implementation = _onedal_jdk_toolchain_impl,
    configure = True,
    local = True,
    environ = [
        "PATH",
        "JAVA_HOME",
    ],
)

def declare_onedal_jdk_toolchain(name):
    onedal_jdk_toolchain(name = name)
    native.register_toolchains("@{}//:all".format(name))
