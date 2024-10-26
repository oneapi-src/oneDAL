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

# Docker Development Environment

## How To Use

There is a simple docker dev environment for the oneDAL development and build process.
It includes dependencies for building all oneDAL components with ``make`` and ``bazel``

Note: The docker setup assumes that it is executed from the oneDAL repo and copies repo files inside the container. In order to build the container locally from the root of the `oneDAL` repository, execute the following:
```shell
docker build -t onedal-dev -f dev/docker/onedal-dev.Dockerfile .
```

Then, in order to use the container interactively, run:
```shell
docker run -it onedal-dev /bin/bash
```
