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
We welcome community contributions to Intel(R) oneAPI Data Analytics Library. You can:

- Submit your changes directly with a [pull request](https://github.com/oneapi-src/oneDAL/pulls).
- Log a bug or feedback with an [issue](https://github.com/oneapi-src/oneDAL/issues).

Refer to our guidelines on [pull requests](#pull-requests) and [isssues](#issues) before you proceed.

## Issues

Use [GitHub issues](https://github.com/oneapi-src/oneDAL/issues) to:
- report an issue
- provide feedback
- make a feature request

**Note**: To report a vulnerability, refer to [Intel vulnerability reporting policy](https://www.intel.com/content/www/us/en/security-center/default.html).

## Pull Requests

To contribute your changes directly to the repository, do the following:
- Make sure you can build the product and run all the examples with your patch.
- Make sure your code is in line with our [coding style](#code-style) as `clang-format` is one of the checks in our public CI.
- For a larger feature, provide a relevant example.
- [Document](#documentation-guidelines) your code.
- [Sign](#sign-your-work) your work.
- [Submit](https://github.com/oneapi-src/oneDAL/pulls) a pull request into the `master` branch.

Public and private CIs are enabled for the repository. Your PR should pass all of our checks. We will review your contribution and, if any additional fixes or modifications are necessary, we may give some feedback to guide you. When accepted, your pull request will be merged into our GitHub* repository.

## Code Style

### ClangFormat

**Prerequisites:** ClangFormat `9.0.0` or later

Our repository contains [clang-format configurations](https://github.com/oneapi-src/oneDAL/blob/master/.clang-format) that you should use on your code. To do this, run:

```
clang-format style=file <your file>
```

Refer to [ClangFormat documentation](https://clang.llvm.org/docs/ClangFormat.html) for more information.

### Coding Guidelines

For your convenience we also added [coding guidelines](http://oneapi-src.github.io/oneDAL/contribution/coding_guide.html) with examples and detailed descriptions of the coding style oneDAL follows. We encourage you to consult them when writing your code.

## Documentation Guidelines

oneDAL uses `Doxygen` for inline comments in public header files that are used to build the API reference and  `reStructuredText` for the Developer Guide. See [oneDAL documentation](https://oneapi-src.github.io/oneDAL/) for reference.

## Sign your work

Use the sign-off line at the end of the patch. Your signature certifies
that you wrote the patch or otherwise have the right to pass it on as an
open-source patch. If you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
Then you add a line to every git commit message:

    Signed-off-by: Kris Smith <kris.smith@email.com>

**Note**: Use your real name.

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

---
**Note:** oneDAL is licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

---
