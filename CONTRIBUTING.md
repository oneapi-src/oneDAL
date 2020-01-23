<!--
******************************************************************************
* Copyright 2014-2020 Intel Corporation
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
We welcome community contributions to Intel(R) oneAPI Data Analytics Library. If you have an idea how to improve the product, you can:

* Let us know about your proposal via [Issues on oneDAL GitHub\*](https://github.com/intel/daal/issues).
* Contribute your changes directly to the repository through [pull request](#pull-requests). 

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- Make sure your code is in line with our [coding style](#code-style).
- For a larger feature, provide a relevant example.
- [Submit](https://github.com/intel/daal/pulls) a pull request into the `master` branch.

Public and private CIs are enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

## Code Style

**Prerequisites:** ClangFormat `9.0.0` or later

Our repository contains [clang-format configurations](https://github.com/intel/daal/blob/master/.clang-format) that you should use on your code. To do this, run:

```
clang-format style=file <your file>
```

Refer to [ClangFormat documentation](https://clang.llvm.org/docs/ClangFormat.html) for more information.

---
**Note:** oneDAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
