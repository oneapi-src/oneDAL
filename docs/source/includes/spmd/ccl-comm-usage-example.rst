.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. highlight:: cpp
.. default-domain:: cpp

::

#include <CL/sycl.hpp>
#include <iomanip>
#include <iostream>

#ifndef ONEDAL_DATA_PARALLEL
#define ONEDAL_DATA_PARALLEL
#endif

#include "oneapi/dal/algo/kmeans.hpp"
#include "oneapi/dal/io/csv.hpp"
#include "oneapi/dal/spmd/ccl/communicator.hpp"

#include "utils.hpp"

namespace dal = oneapi::dal;

int main(int argc, char const *argv[]) {
   int status = MPI_Init(nullptr, nullptr);
   if (status != MPI_SUCCESS) {
      throw std::runtime_error{ "Problem occurred during MPI init" };
   }

   auto device = sycl::gpu_selector{}.select_device();
   std::cout << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
   sycl::queue queue{ device };
   run(q);
   // Create algorithm descriptor
   const auto kmeans_desc = dal::kmeans::descriptor<>()
                                 .set_cluster_count(20)
                                 .set_max_iteration_count(5)
                                 .set_accuracy_threshold(0.001);
   // Create communicator using MPI backend
   auto comm = dal::preview::spmd::make_communicator<dal::preview::spmd::backend::ccl>(queue);
   // Get rank id and rank count
   auto rank_id = comm.get_rank();
   auto rank_count = comm.get_rank_count();

   // Load data files per rank and prepare local_input variable of type
   // dal::kmeans::train_input
   ...
   // Run distrubuted computation
   const auto result_train = dal::preview::train(comm, kmeans_desc, local_input);
   // Report results on zero rank
   if(comm.get_rank() == 0) {
      std::cout << "Iteration count: " << result_train.get_iteration_count() << std::endl;
      std::cout << "Objective function value: " << result_train.get_objective_function_value()
               << std::endl;
      std::cout << "Centroids:\n" << result_train.get_model().get_centroids() << std::endl;
   }

   status = MPI_Finalize();
   if (status != MPI_SUCCESS) {
      throw std::runtime_error{ "Problem occurred during MPI finalize" };
   }
   return 0;
}