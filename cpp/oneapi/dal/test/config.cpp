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

#include "oneapi/dal/test/config.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "oneapi/dal/test/common.hpp"
#include "oneapi/dal/backend/interop/common_dpc.hpp"
#endif

namespace oneapi::dal::test {

#ifdef ONEDAL_DATA_PARALLEL

static sycl::queue get_default_queue() {
    return sycl::gpu_selector{};
}

void global_setup() {
    dal::backend::interop::enable_daal_sycl_execution_context_cache();
    test_queue_provider::get_instance().init(get_default_queue());
}

void global_cleanup() {
    dal::backend::interop::cleanup_daal_sycl_execution_context_cache();
    test_queue_provider::get_instance().reset();
}

#else
void global_setup() {}
void global_cleanup() {}
#endif

} //namespace oneapi::dal::test
