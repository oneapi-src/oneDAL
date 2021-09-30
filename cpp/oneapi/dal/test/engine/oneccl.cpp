/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifdef ONEDAL_DATA_PARALLEL

#include "oneapi/dal/test/engine/spmd.hpp"
#include "oneapi/dal/detail/oneccl/communicator.hpp"
#include <memory>

namespace oneapi::dal::test::engine {

std::unique_ptr<dal::detail::spmd_communicator> global_oneccl_communicator;

dal::detail::spmd_communicator get_oneccl_communicator(const sycl::queue& queue) {
    if (!global_oneccl_communicator) {
        global_oneccl_communicator.reset(new dal::detail::oneccl_communicator{ queue });
    }
    return *global_oneccl_communicator;
}

class oneccl_communicator_global_setup : public global_setup_action {
public:
    void init(const global_config& config) override {
        // TODO: Pass argc/argv to the init via global config?
        ccl::init();
        const int status = MPI_Init(nullptr, nullptr);
        if (status != MPI_SUCCESS) {
            throw std::runtime_error{ "Problem occurred during MPI init" };
        }
    }

    void tear_down() override {
        global_oneccl_communicator.reset(nullptr);
        const int status = MPI_Finalize();
        if (status != MPI_SUCCESS) {
            throw std::runtime_error{ "Problem occurred during MPI finalize" };
        }
    }
};

REGISTER_GLOBAL_SETUP(oneccl_communicator, oneccl_communicator_global_setup)

} // namespace oneapi::dal::test::engine
#endif
