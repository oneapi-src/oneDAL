# Bazel Guide
## Install Bazel on Linux
1. Download Bazel 4.0.0
   ```sh
   wget -O bazel https://github.com/bazelbuild/bazel/releases/download/4.0.0/bazel-4.0.0-linux-x86_64
   ```
   > Note: If you are using proxy don't forget to set
   `http_proxy` and `https_proxy` environment variables

2. Put it somewhere on **local disk** (use of remote shared
   drives is not recommended) and set executable attributes.
   ```sh
   mv bazel <my-user-dir-utils>/bin
   chmod +x bazel
   ```

3. Create Bazel cache directory on **local disk** and
   configure path to the cache in `~/.bazelrc`.
   ```sh
   mkdir -p <my-user-dir-on-local-disk>/.bazel-cache
   echo "startup --output_user_root=<my-user-dir-on-local-disk>/.bazel-cache" > ~/.bazelrc
   ```

4. Add `bazel` to the `$PATH`.
   ```sh
   export PATH=<my-user-dir-on-local-disk>/bin:$PATH
   bazel --version # Should be "bazel 4.0.0"
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

## Common Bazel commands
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
  you can simply use `cpp/oneapi/dal:array_test`.

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

- `clean` removes built artifacts and cleans the cache.
  ```sh
  bazel clean
  ```

  When you update Bazel version, want to shutdown Bazel server or suspect that
  something went wrong it is recommended to add `--expunge` option.
  ```sh
  bazel clean --expunge
  ```

## Bazel options
- `--compilation_mode [-c]` Bazel comes with three compilation modes `opt`,
  `dbg` and `fastbuild`. \
  Possible values:
   - `opt` _(default)_ compiles everything with optimizations `-O2`.
   - `dbg` enables `-g`, `-O0` compiler switches and **assertions**.
   - `fastbuild` optimizes build time, no optimizations, no debug information.
     Useful when one introduces massive changes and wants to check whether they
     break the build.

   Example:
   ```sh
   bazel test -c dbg //cpp/oneapi/dal:tests
   ```

   It is highly encouraged to run newly added tests on the local machine with
   `-c dbg` option to test assertions.

- `--config` Takes effect only for `run` or `test` commands. Specifies
  interface that is used to build and run tests. \
  Possible values:
  - `<not specified>` _(default)_ Build and run all tests.
  - `host` Build and run tests for HOST part without dependencies on DPC++
    runtime. Uses regular C++ compiler.
  - `dpc` Build and run tests for DPC++ part. Uses Intel(R) DPC++ Compiler.
  - `public` Build and run tests for public C++ and DPC++ parts.
  - `host-public` Build and run tests only for public part of C++ interfaces.
  - `dpc-public` Build and run tests only for public part of DPC++ interfaces.
  - `private` Build and run tests for private C++ and DPC++ parts.
  - `host-private` Build and run tests only for private C++ part.
  - `dpc-private` Build and run tests only for private DPC++ part.

   Example:
   ```sh
   # To run all HOST tests
   bazel test --config=host //cpp/oneapi/dal:tests

   # To run tests for internal DPC++ functionality
   bazel test --config=dpc-private //cpp/oneapi/dal:tests
   ```

- `--device` Takes effect only for `run` or `test` commands and option
  `--config` that implies build of DPC++ interfaces. Specifies device to create
  a queue and run tests. \
  Possible values:
  - `auto` _(default)_ Automatically detects available device. Tries to detect
    GPU first, if no GPUs available tries to use CPU. If no device runtimes
    found, uses `host_selector`.
  - `cpu` Creates queue using `cpu_selector`.
  - `gpu` Creates queue using `gpu_selector`.

   Example:
   ```sh
   bazel test --config=dpc --device=gpu //cpp/oneapi/dal:tests
   ```

- `--cpu` CPU instruction sets to compile library for. \
  Possible values:
  - `auto` _(default)_ Automatically detects highest available instruction set
    on the local machine. If detection failed, uses `avx2`.
  - `modern` Compiles for `sse2`, `avx`, `avx2`, `avx512`.
  - `all` Compiles for all instruction sets listed below.
  - Any comma-separated combination of the following values:
    - `sse2`
    - `ssse3`
    - `sse42`
    - `avx`
    - `avx2`
    - `avx512`
    - `avx512_mic`

   Example:
   ```sh
   bazel test --cpu="avx,avx512" //cpp/oneapi/dal:tests
   ```

- `--test_external_datasets` A switch that enables
  [tests which depend on datasets stored on local drive](#tests-that-use-external-datasets).
  Disabled by default.

   Example:
   ```sh
   bazel test --test_external_datasets //cpp/oneapi/dal:tests
   ```

- `--test_nightly` A switch that enables [nightly tests](#nightly-tests).
  Disabled by default.

   Example:
   ```sh
   bazel test --test_nightly //cpp/oneapi/dal:tests
   ```

- `--test_weekly` A switch that enables [weekly tests](#weekly-tests). Nightly
  tests are included to weekly. Disabled by default.

   Example:
   ```sh
   bazel test --test_weekly //cpp/oneapi/dal:tests
   ```

- `--test_link_mode` Specifies linking mode for tests. \
  Possible values:
  - `dev` _(default)_ Automatically determines the set of object files need to
    be linked to the particular test.
  - `release_static` Links tests against static libraries found in `$DAALROOT`.
  - `release_dynamic` Links tests against dynamic libraries found in `$DAALROOT`.

   Example:
   ```sh
   export DAALROOT=`pwd`/__release_lnx/daal/latest
   bazel test --test_link_mode=release_dynamic //cpp/oneapi/dal:tests
   ```

- `--test_thread_mode` Specifies threading mode for tests. \
  Possible values:
  - `par` _default_ Links against `onedal_thread`.
  - `seq` Links against `onedal_sequential`.

   Example:
   ```sh
   bazel test --test_thread_mode=seq //cpp/oneapi/dal:tests
   ```

- `--test_disable_fp64` A switch that disables tests for double precision (`fp64`).

  Example:
  ```sh
  bazel test --test_disable_fp64 //cpp/oneapi/dal/algo/pca:tests
  ```

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
  bazel test //cpp/oneapi/dal:tests
  ```

  This will run all tests including HOST and DPC++ interfaces. In case of DPC++
  tests, queue will be created depending machine configuration. If there is a
  GPU, `sycl::gpu_selector` is used, otherwise queue is created with
  `sycl::cpu_selector`.

- To run all HOST tests use the following commands:
  ```sh
  bazel test --config=host //cpp/oneapi/dal:tests
  ```

- To run all DPC++ tests use the following commands:
  ```sh
  bazel test --config=dpc //cpp/oneapi/dal:tests
  ```

  If you need run DPC++ tests on specific device use the `device` option:
  ```sh
  bazel test --config=dpc --device=gpu //cpp/oneapi/dal:tests
  ```

- To run specific **set of tests**, e.g., for specific algorithm:
  ```sh
  bazel test //cpp/oneapi/dal/algo/svm:tests
  ```

  This will run all tests for SVM algorithm including HOST and DPC++ interfaces.
  To control type of interfaces and target device, use the `--config` and
  `--device` options described above.

- To run specific **test** and see output in stdout:
  ```sh
  # Use HOST interface to build and run the test
  bazel run //cpp/oneapi/dal/algo/pca:test_batch_host

  # Use DPC++ interface to build and run the test
  bazel run //cpp/oneapi/dal/algo/pca:test_batch_dpc
  ```

  This is useful when one tests is failed on `bazel test` and you want to focus
  on one particular test to fix the problem. The name of the test can be taken
  from `bazel test //cpp/oneapi/dal:tests` output as it prints names of all test
  targets.

- You can run test executables produced by Bazel manually. The compiled
  executables are stored in corresponding `bazel-bin` subdirectory. For example,
  if the test target name is `//cpp/oneapi/dal/pca:test_batch_host`, the
  executable `test_batch_host` will be stored in
  `<repo-root>/bazel-bin/cpp/oneapi/dal/algo/pca` directory.

## Advanced testing
### Tests that use external datasets
Some tests may depend on quite large datasets that cannot be stored in the Git
repository. In that case test should read dataset from `$DAAL_DATASETS`
directory that could point to datasets storage on local disk. We call such
datasets _external_.

Such test cases are disabled by default, but can be enabled using
`--test_external_datasets` switch.
```sh
bazel test --test_external_datasets //cpp/oneapi/dal:tests
```

Test cases with external datasets must be marked by `[external-dataset]` tag,
which is used to filter out tests at runtime if no `--test_external_datasets`
provided.
```c++
TEST("my test that uses external dataset", "[external-dataset]") {
    // Code that reads some external dataset
}
```

### Nightly tests
Nightly tests are disabled by default. The `--test_nightly` option enables tests
marked with `[nightly]` tags.
```sh
bazel test --test_nightly //cpp/oneapi/dal:tests
```

Mark nightly test cases with `[nightly]` tag:
```c++
TEST("my test that takes long time", "[nightly]") { /* ... */ }
```

### Weekly tests
Weekly tests are disabled by default. The `--test_weekly` option enables tests
marked with `[weekly]` or `[nightly]` tags. **Weekly tests include nightly.**
```sh
bazel test --test_weekly //cpp/oneapi/dal:tests
```

Mark weekly test cases with `[weekly]` tag:
```c++
TEST("my test that takes very long time", "[weekly]") { /* ... */ }
```

### Tests for public interface
Sometimes it is necessary to build and run only part of tests responsible for
public interface testing, for example, to test backward compatibility. There is
the `host-public` config designed for that use case.
```sh
bazel test --config=host-public //cpp/oneapi/dal:tests
```

Test is categorized to public or private using the `private` flag of the
`dal_tst_suite` rule. By default, all test suites are public, if the new test
suite for the private functionality is defined, the flag should be turned on.
```py
dal_test_suite(
    name = "my_tests",
    private = True,
    ...
)
```

### Run tests for the existing oneDAL build
1. Set `DAALROOT` env var:
   ```sh
   export DAALROOT=`pwd`/__release_lnx/daal/latest
   ```

2. Run Bazel:
   - If linking against static version of the lib is required:
     ```sh
     bazel test --test_link_mode=release_static //cpp/oneapi/dal:tests
     ```

   - If linking against dynamic version of the lib is required:
     ```sh
     bazel test --test_link_mode=release_dynamic //cpp/oneapi/dal:tests
     ```

## What is missing in this guide
- How to get make-like release structure
