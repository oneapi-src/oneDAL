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

#include "oneapi/dal/network/network.hpp"
#include <mpi.h>
#include <algorithm>
#include <vector>

namespace oneapi::dal::network::mpi {


namespace detail {

class communicator: public oneapi::dal::network::detail::communicator_base {
public:
    void allreduce(float* ptr, size_t n) override {
        // TODO: ISA specific things?
        std::vector<float> v(n);
        std::copy(ptr, ptr + n, v.begin());
        MPI_Allreduce(v.data(), ptr, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    }

    void allreduce(double* ptr, size_t n) override {
        std::vector<double> v(n);
        std::copy(ptr, ptr + n, v.begin());
        MPI_Allreduce(v.data(), ptr, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    void allreduce(int* ptr, size_t n) override {
        std::vector<int> v(n);
        std::copy(ptr, ptr + n, v.begin());
        MPI_Allreduce(v.data(), ptr, n, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
};

}

class network: public oneapi::dal::network::network {
public:
    std::shared_ptr<oneapi::dal::network::detail::communicator_base> get_communicator() const override
    {
        return std::shared_ptr<oneapi::dal::network::detail::communicator_base>(new detail::communicator() );
    }
};

}
