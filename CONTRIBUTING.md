<!--
******************************************************************************
* Copyright 2014 Intel Corporation
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

# How to Contribute
We welcome community contributions to Intel(R) oneAPI Data Analytics Library. You can:

- Submit your changes directly with a [pull request](https://github.com/oneapi-src/oneDAL/pulls).
- Log a bug or make a feature request with an [issue](https://github.com/oneapi-src/oneDAL/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [issues](#issues) before you proceed.

## Contacting maintainers
You may reach out to Intel project maintainers privately at onedal.maintainers@intel.com.
Codeoners configuration define specific maintainers for coresponding code sections, however currently limited to Intel members only at this point. With futher migration to UXL we will be fixing this, but here are non Intel contacts: 
For ARM specifics you may contact: @rakshithgb-fujitsu
For RISC-V specifics you may contact: @keeranroth

## Issues

Use [GitHub issues](https://github.com/oneapi-src/oneDAL/issues) to:
- report an issue
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- Make sure your code is in line with our [coding style](#code-style) as `clang-format` is one of the checks in our public CI.
- For a larger feature, provide a relevant example.
- [Document](#documentation-guidelines) your code.
- [Submit](https://github.com/oneapi-src/oneDAL/pulls) a pull request into the `main` branch.

Public and private CIs are enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

## Code Style

### ClangFormat

**Prerequisites:** ClangFormat `9.0.0` or later

Our repository contains [clang-format configurations](https://github.com/oneapi-src/oneDAL/blob/main/.clang-format) that you should use on your code. To do this, run:

```
clang-format style=file <your file>
```

Refer to [ClangFormat documentation](https://clang.llvm.org/docs/ClangFormat.html) for more information.

### Coding Guidelines

For your convenience we also added [coding guidelines](http://oneapi-src.github.io/oneDAL/contribution/coding_guide.html) with examples and detailed descriptions of the coding style oneDAL follows. We encourage you to consult them when writing your code.

## Documentation Guidelines

oneDAL uses `Doxygen` for inline comments in public header files that are used to build the API reference and  `reStructuredText` for the Developer Guide. See [oneDAL documentation](https://oneapi-src.github.io/oneDAL/) for reference.

---
**Note:** oneDAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
