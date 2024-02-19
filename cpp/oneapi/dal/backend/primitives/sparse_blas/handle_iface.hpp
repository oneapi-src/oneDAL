/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <mkl_dal_sycl.hpp>

namespace oneapi::dal::backend::primitives {

namespace mkl = oneapi::fpk;

#ifdef ONEDAL_DATA_PARALLEL
///
class sparse_matrix_handle_iface {
public:
    sparse_matrix_handle_iface(sycl::queue& queue);

    virtual ~sparse_matrix_handle_iface();

    mkl::sparse::matrix_handle_t handle;

private:
    sycl::queue& queue_;
};

#endif // ifdef ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::backend::primitives
