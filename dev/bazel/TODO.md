### TODO:
- [ ] **Windows support.** Bazel shall provide toolchain implementation for
  Windows.

- [ ] **Release to oneAPI structure.** Bazel shall write headers, binaries,
  examples and scripts to oneAPI release structure as the current make does.

- [ ] **Extend compiler support matrix.** Current status:
  |         |        Intel       |        DPC++       |         GCC        |       Clang        |        MSVC        |
  |---------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
  | Linux   |        :x:         |         :x:        | :heavy_check_mark: |                    |                    |
  | MacOs   |        :x:         |         :x:        |                    |        :x:         |                    |
  | Windows |        :x:         |         :x:        |                    |                    |        :x:         |

- [ ] **Automatic makefile generation for examples on Linux/MacOs.** Bazel
  shall generate Makefile for examples when release build is required.

- [ ] **Automatic  VS solution generation for examples on Windows.** Bazel shall
  generate Visual Studio solution for examples then release build is required.

- [ ] **Automatic host architecture identification.** Bazel shall detect host
  machine architecture and configure best CPU id automatically.

- [ ] **Toolchain code unification.** There is logic duplication for toolchain
  configuration on Linux/MacOs/Windows.

