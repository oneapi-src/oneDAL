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

#include "oneapi/dal/test/engine/spmd.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"

namespace oneapi::dal::test::engine {

spmd::communicator<spmd::device_memory_access::none> get_global_mpi_host_communicator() {
    return dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>();
}

#ifdef ONEDAL_DATA_PARALLEL
spmd::communicator<spmd::device_memory_access::usm> get_global_mpi_device_communicator(
    sycl::queue& queue) {
    return dal::preview::spmd::make_communicator<dal::preview::spmd::backend::mpi>(queue);
}
#endif

class mpi_communicator_global_setup : public global_setup_action {
public:
    void init(const global_config& config) override {
        // TODO: Pass argc/argv to the init via global config?
        const int status = MPI_Init(nullptr, nullptr);
        if (status != MPI_SUCCESS) {
            throw std::runtime_error{ "Problem occurred during MPI init" };
        }
    }

    void tear_down() override {
        const int status = MPI_Finalize();
        if (status != MPI_SUCCESS) {
            throw std::runtime_error{ "Problem occurred during MPI finalize" };
        }
    }
};

REGISTER_GLOBAL_SETUP(mpi_communicator, mpi_communicator_global_setup)

} // namespace oneapi::dal::test::engine
