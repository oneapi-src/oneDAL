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

#pragma once

#ifdef ONEAPI_DAL_DATA_PARALLEL

#include <CL/sycl.hpp>

namespace oneapi::dal::detail {

template <typename T>
class dp_default_delete {
public:
    explicit dp_default_delete(const sycl::queue& queue)
        : queue_(queue) {}

    void operator()(T* data) {
        sycl::free(data, queue_);
    }

private:
    sycl::queue queue_;
};

} // namespace oneapi::dal::detail


#endif // ONEAPI_DAL_DATA_PARALLEL
