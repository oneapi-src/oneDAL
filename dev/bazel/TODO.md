<!--
******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

### TODO:
- [ ] **Windows support.** Bazel shall provide toolchain implementation for
  Windows.

- [ ] **Release to oneAPI structure.** Bazel shall write headers, binaries,
  examples and scripts to oneAPI release structure as the current make does.

- [ ] **Extend compiler support matrix.** Current status:
  |         |        Intel       |        DPC++       |         GCC        |       Clang        |        MSVC        |
  |---------|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
  | Linux   |        :x:         |         :x:        | :heavy_check_mark: |                    |                    |
  | Windows |        :x:         |         :x:        |                    |                    |        :x:         |

- [ ] **Automatic host architecture identification.** Bazel shall detect host
  machine architecture and configure best CPU id automatically.

- [ ] **Toolchain code unification.** There is logic duplication for toolchain
  configuration on Linux/Windows.

