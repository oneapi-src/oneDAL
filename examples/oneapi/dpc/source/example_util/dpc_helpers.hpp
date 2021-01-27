/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
*******************************************************************************/

#pragma once

#include <vector>
#include <CL/sycl.hpp>

template <typename Selector>
void try_add_device(std::vector<sycl::device>& devices) {
    try {
        devices.push_back(Selector{}.select_device());
    }
    catch (...) {
    }
}

std::vector<sycl::device> list_devices() {
    std::vector<sycl::device> devices;
    try_add_device<sycl::host_selector>(devices);
    try_add_device<sycl::cpu_selector>(devices);
    try_add_device<sycl::gpu_selector>(devices);
    return devices;
}
