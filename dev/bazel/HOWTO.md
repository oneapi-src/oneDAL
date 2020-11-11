# Bazel Guide
## Install Bazel on Linux
1. Download Bazel 3.7.0
   ```sh
   wget -O bazel https://github.com/bazelbuild/bazel/releases/download/3.7.0/bazel-3.7.0-linux-x86_64
   ```
   > Note: If you are using proxy don't forget to set
   `http_proxy` and `https_proxy` environment variables

2. Put it somewhere on **local disk** (use of remote shared
   drives is not recommended) and set executable attributes.
   ```sh
   mv bazel <my-user-dir-utils>/bin
   chmod +x bazel
   ```

3. Create bazel cache directory on **local disk** and
   configure path to the cache in `~/.bazelrc`.
   ```sh
   mkdir -p <my-user-dir-utils>/.bazel-cache
   echo "startup --output_user_root=<my-user-dir-utils>/.bazel-cache" > ~/.bazelrc
   ```

4. Add `bazel` to the `$PATH`.
   ```sh
   export PATH=<my-user-dir-utils>/bin:$PATH
   bazel --version # Should be "bazel 3.7.0"
   ```

## Build options
### Debug vs Release mode
Bazel comes with three build modes `opt`, `dbg` and `fastbuild`.
- `opt` compiles everything with optimizations `-O2`. **This is default.**
- `dbg` enables `-g`, `-O0` compiler switches and **enables assertions**.
- `fastbuild` optimizes build time, no optimizations, no debug information.
  Useful when one introduces massive changes and wants to check whether
  they break the build.

One of three build modes can be used together with Bazel commands
described bellow.
```sh
bazel <bazel-command> -c dbg <target-names>
```

### Compiler choice
Be default our build system configures Bazel to use Intel(R) C++ Compiler
in case of normal C++ code and Intel(R) oneAPI DPC++ Compiler in case of
DPC++. If Intel(R) C++ Compiler is not available in the `$PATH`, Bazel
tries to find default compiler for specific OS, e.g., GCC for Linux.

The C++ compiler can be forcibly changed using environment variable `CC`.
```sh
export CC=gcc
bazel <bazel-command> ... # Will use GCC for normal C++ code
```

## Common Bazel command
The most used Bazel commands are `build`, `test` and `run`.
- `build` builds specified target.
  ```sh
  bazel build <target-name>
  ```

  Target names are always bound to the location in repository
  and defined in `BUILD` files. For example, test for `dal::array`
  is defined in `<repo-root>/cpp/oneapi/dal/BUILD`, so the name of
  the Bazel target will be `//cpp/oneapi/dal:array_test`.
  Here `//` stands for repository root.

  For simplicity, you can use relative target names.
  For example, if you are already in the repository root,
  you can simply use `cpp/oneapi/dal:array_test`, or
  ```sh
  cd cpp/oneapi
  bazel build dal:array_test
  ```

- `run` builds and runs specified target
  ```sh
  bazel run <target-name>
  ```

  > Note: There is no need to build and run targets separately,
  `run` builds if needed.

  > Note: `run` cannot be used to run multiple targets at once,
  consider use of `test`.

- `test` builds and runs specified test target
  ```sh
  bazel test <test-target-name>
  ```

  There is no syntax difference between in targets and test targets,
  but test targets may contain a set of multiple executables
  called *test suite*.


## Build recipes for oneDAL
### Run oneAPI examples
- To run all oneAPI C++ example use the following commands:
```sh
bazel test //examples/oneapi/cpp:all
```

- To run all oneAPI DPC++ examples ... It's not implemented yet...

### Run oneAPI tests
- To run all test use the following commands:
  ```sh
  bazel test //cpp/oneapi/dal:tests      # Runs all C++ tests
  bazel test //cpp/oneapi/dal:tests_dpc  # Runs all DPC++ tests
  bazel test //cpp/oneapi/dal:all        # Runs both C++ and DPC++
  ```

- To run specific set of tests, e.g., for specific algorithm:
  ```sh
  bazel test //cpp/oneapi/dal/algo/svm:tests     # For C++
  bazel test //cpp/oneapi/dal/algo/svm:tests_dpc # For DPC++
  bazel test //cpp/oneapi/dal/algo/svm:all       # For both C++ and DPC++
  ```

- To run specific test and see output in stdout:
  ```sh
  bazel run //cpp/oneapi/dal/algo/svm:<specific_test_name>
  ```

  This is useful when one tests is failed on `bazel test` and
  you want to focus on one particular test to fix the problem. The
  name of the test can be taken from `bazel test //cpp/oneapi/dal:tests`
  output as it prints names of all test targets.

- You can run test executables produced by Bazel manually. The
  compiled executables are stored in corresponding `bazel-bin` subdirectory.
  For example, if the test target name is `//cpp/oneapi/dal/table:common_test`,
  the executable `common_test` will be stored in `<repo-root>/bazel-bin/cpp/oneapi/dal/table`
  directory.


### What is missing in this guide
- Choice of threading layer
- How to get make-like release structure
- How to use binaries build by make to run Bazel tests
