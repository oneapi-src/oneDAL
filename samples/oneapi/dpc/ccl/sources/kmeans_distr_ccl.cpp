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

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/spmd/ccl/communicator.hpp"


#include "utils.hpp"

using namespace std;
using namespace sycl;

inline void mpi_finalize() {
    int is_finalized = 0;
    MPI_Finalized(&is_finalized);

    if (!is_finalized)
        MPI_Finalize();
}

int main(int argc, char const *argv[]) {
    ccl::init();
    int status = MPI_Init(nullptr, nullptr);


    auto device = gpu_selector{}.select_device();
    cout << "Running on " << device.get_info<info::device::name>() << "\n";
    queue q{ device };
        // const size_t count = 10 * 1024 * 1024;

    int size = 0;
    int rank = 0;


    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    atexit(mpi_finalize);

    /* create kvs */
    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type main_addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        main_addr = kvs->get_address();
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(main_addr);
    }

    /* create communicator */
    auto dev = ccl::create_device(q.get_device());
    auto ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, dev, ctx, kvs);

    /* create stream */
    auto stream = ccl::create_stream(q);

    /* create buffers */
    const size_t granularity = 10;
    const size_t rank_count = size;

    std::vector<size_t> recv_counts(rank_count); // size_t instead std::int64_t
    std::vector<size_t> displs(rank_count);
    size_t total_size = 0;
    constexpr size_t empty_rank = 1;
    for (size_t i = 0; i < rank_count; i++) {
        recv_counts[i] = i != empty_rank ? (i + 1) * granularity : 0;
        displs[i] = total_size;
        total_size += recv_counts[i];
    }

    std::vector<float> check_buf(total_size);
    float *recv_buffer = malloc_device<float>(total_size, q);

    const size_t data_type_size = sizeof(float);
    std::vector<void*> recv_bufs(rank_count);
    for (size_t i = 0; i < rank_count; i++) {
        recv_bufs[i] =  recv_buffer + data_type_size * displs[i];
    }

    std::vector<float> final_buffer(total_size);
    std::int64_t offset = 0;
    for (size_t i = 0; i < rank_count; i++) {
        for (size_t j = 0; j < recv_counts[i]; j++) {
            final_buffer[offset] = float(i);
            offset++;
        }
    }
    const size_t rank_size = recv_counts[rank];
    float *send_buffer = sycl::malloc_device<float>(rank_size, q);
    auto e = q.submit([&](auto &h) {
            h.parallel_for(rank_size , [=](auto id) {
                    send_buffer[id] = float(rank);
            });
            });

        ccl::allgatherv(send_buffer, rank_size, recv_bufs, recv_counts, ccl::datatype::float32, comm, stream).wait();
        q.submit([&](handler &h){
            h.memcpy(check_buf.data(), recv_buffer, sizeof(float) * total_size);
        });
        
 


    status = MPI_Finalize();
    if (status != MPI_SUCCESS) {
        throw std::runtime_error{ "Problem occurred during MPI finalize" };
    }
    int flag = 0;
    for (size_t i = 0; i < total_size; i++) {
        std::cout<< i << "check_buf[i] = "<< check_buf[i] << " final_buffer[i] =  " << final_buffer[i] << "\n";
        if (check_buf[i] == final_buffer[i]){
            flag = 0;
        }
        else{
            flag = 1;
             throw std::runtime_error{ "Problem with check_buf" };
        }
    }
    return flag;
}
