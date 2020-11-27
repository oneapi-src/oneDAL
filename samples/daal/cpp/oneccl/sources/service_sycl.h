/* file: service_sycl.h */
/*******************************************************************************
* Copyright 2020 Intel Corporation
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

/*
!  Content:
!    Auxiliary sycl functions used in C++ examples
!******************************************************************************/

#ifndef _SERVICE_SYCL_H
#define _SERVICE_SYCL_H

#include <list>
#include <memory>

#include <CL/cl.h>
#include <CL/sycl.hpp>

#include "service.h"

std::vector<sycl::device> get_gpus() {
    auto platforms = sycl::platform::get_platforms();
    for (auto p : platforms) {
        auto devices = p.get_devices(sycl::info::device_type::gpu);
        if (!devices.empty()) {
            return devices;
        }
    }
    return {};
}

#endif
