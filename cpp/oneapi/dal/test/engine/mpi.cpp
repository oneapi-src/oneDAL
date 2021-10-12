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
#include "oneapi/dal/detail/mpi/communicator.hpp"

namespace oneapi::dal::test::engine {

std::unique_ptr<dal::detail::spmd_communicator> global_mpi_communicator;

dal::detail::spmd_communicator get_global_mpi_communicator() {
    if (!global_mpi_communicator) {
        global_mpi_communicator.reset(new dal::detail::mpi_communicator{ MPI_COMM_WORLD });
    }
    return *global_mpi_communicator;
}

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
        global_mpi_communicator.reset();
        const int status = MPI_Finalize();
        if (status != MPI_SUCCESS) {
            throw std::runtime_error{ "Problem occurred during MPI finalize" };
        }
    }
};

REGISTER_GLOBAL_SETUP(mpi_communicator, mpi_communicator_global_setup)

} // namespace oneapi::dal::test::engine
