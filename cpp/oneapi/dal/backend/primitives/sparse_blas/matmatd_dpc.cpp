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

#include "oneapi/dal/backend/primitives/blas/misc.hpp"
#include "oneapi/dal/backend/primitives/sparse_blas/matmatd.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Float, ndorder co>
sycl::event matmatd(sycl::queue &queue,
                    transpose transpose_a,
                    transpose transpose_b,
                    const Float alpha,
                    sparse_matrix_handle& a,
                    sparse_matrix_handle& b,
                    const Float beta,
                    ndview<Float, 2, co>& c,
                    const std::vector<sycl::event> &dependencies) {
    ONEDAL_ASSERT(c.has_mutable_data());

    if (co == ndorder::c) {
        return mkl::sparse::matmatd(queue,
                                    order_as_layout(co),
                                    transpose_to_mkl(transpose_a),
                                    transpose_to_mkl(transpose_b),
                                    alpha,
                                    dal::detail::get_impl(a).get(),
                                    dal::detail::get_impl(b).get(),
                                    beta,
                                    c.get_mutable_data(),
                                    c.get_dimension(0),
                                    c.get_dimension(1),
                                    c.get_leading_stride(),
                                    dependencies);
    }

    return sycl::event();
}

} // namespace oneapi::dal::backend::primitives
