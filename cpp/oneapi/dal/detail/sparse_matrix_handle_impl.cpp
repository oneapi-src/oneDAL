/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/detail/sparse_matrix_handle_impl.hpp"

namespace oneapi::dal::detail {

namespace v1 {

#ifdef ONEDAL_DATA_PARALLEL

sparse_matrix_handle_impl::sparse_matrix_handle_impl(sycl::queue& queue) : queue_(queue) {
    mkl::sparse::init_matrix_handle(&handle_);
}

sparse_matrix_handle_impl::~sparse_matrix_handle_impl() {
    mkl::sparse::release_matrix_handle(queue_, &handle_, {}).wait();
}

#endif // ONEDAL_DATA_PARALLEL

} // namespace v1
} // namespace oneapi::dal::detail
