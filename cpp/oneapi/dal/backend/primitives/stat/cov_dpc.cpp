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

#include "oneapi/dal/backend/primitives/stat/cov.hpp"
#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/loops.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

#include <CL/sycl/ONEAPI/experimental/builtins.hpp>

namespace oneapi::dal::backend::primitives {

template <typename Float>
sycl::event correlation(sycl::queue& queue,
                        const table& data,
                        const ndview<Float, 1>& sums,
                        ndview<Float, 2>& corr,
                        ndview<Float, 1>& means,
                        ndview<Float, 1>& vars,
                        ndview<Float, 1>& tmp,
                        const event_vector& deps) {
    ONEDAL_ASSERT(data.has_data());
    ONEDAL_ASSERT(sums.has_data());
    ONEDAL_ASSERT(corr.has_mutable_data());
    ONEDAL_ASSERT(means.has_mutable_data());
    ONEDAL_ASSERT(vars.has_mutable_data());
    ONEDAL_ASSERT(tmp.has_mutable_data());
    ONEDAL_ASSERT(corr.get_shape(0) == corr.get_shape(1), "Correlation matrix must be square");
    ONEDAL_ASSERT(corr.get_shape(0) == data.get_column_count(),
                  "Dimensions of correlation matrix must match column count of input data");
    ONEDAL_ASSERT(sums.get_shape(0) == data.get_column_count(),
                  "Element count of sums must match column count of input data");
    ONEDAL_ASSERT(vars.get_shape(0) == data.get_column_count(),
                  "Element count of vars must match column count of input data");
    ONEDAL_ASSERT(means.get_shape(0) == data.get_column_count(),
                  "Element count of means must match column count of input data");
    ONEDAL_ASSERT(tmp.get_shape(0) == data.get_column_count(),
                  "Element count of temporary buffer must match column count of input data");

    // TODO: Determine optimal block size
    constexpr std::int64_t block_max_row_count = 4096;

    sycl::event gemm_event;
    array<Float> data_block_flat;
    const auto data_acc = row_accessor<const Float>{ data };

    for_each_block(data, block_max_row_count, [&](const row_block_info& bi) mutable {
        data_acc.pull(queue, data_block_flat, bi.get_row_range());
        const auto x = ndview<Float, 2>::wrap(data_block_flat.get_data(), bi.get_shape());

        // For the first block, call C = X^T x X, beta = 0.0
        // For the next blocks, call C = X^T x X + C, beta = 1.0
        const Float beta = Float(bi.get_block_index() != 0);
        gemm_event = gemm(queue, x.t(), x, corr, Float(1), beta, { gemm_event });
    });

    const std::int64_t n = data.get_row_count();
    const std::int64_t p = data.get_column_count();
    const Float inv_n = Float(1.0 / double(n));
    const Float inv_n1 = (n > 1) ? Float(1.0 / double(n - 1)) : 1;

    const Float* sums_ptr = sums.get_data();
    Float* corr_ptr = corr.get_mutable_data();
    Float* means_ptr = means.get_mutable_data();
    Float* vars_ptr = vars.get_mutable_data();
    Float* tmp_ptr = tmp.get_mutable_data();

    auto mean_var_event = queue.submit([&](sycl::handler& cgh) {
        // TODO: Get information about local size from device
        const auto range = make_multiple_nd_range_1d(p, 256);
        const auto p_int = dal::detail::integral_cast<int>(p);

        cgh.depends_on(gemm_event);
        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const int i = id.get_global_id();
            if (i < p) {
                const Float s = sums_ptr[i];
                const Float m = inv_n * s * s;
                const Float c = corr_ptr[i * p_int + i];

                means_ptr[i] = s * inv_n;
                vars_ptr[i] = inv_n1 * (c - m);
                tmp_ptr[i] = sycl::sqrt(c);
            }
        });
    });

    auto finalize_corr_event = queue.submit([&](sycl::handler& cgh) {
        // TODO: Get information about local size from device
        const auto range = make_multiple_nd_range_1d(p * p, 256);
        const auto p_int = dal::detail::integral_cast<int>(p);

        cgh.depends_on(mean_var_event);
        cgh.parallel_for(range, [=](sycl::nd_item<1> id) {
            const int gi = id.get_global_id();
            if (gi < p_int * p_int) {
                const int i = gi / p_int;
                const int j = gi % p_int;
                const Float is_diag = Float(i == j);

                Float c = corr_ptr[gi];
                c -= inv_n * sums_ptr[i] * sums_ptr[j];
                c /= tmp_ptr[i] * tmp_ptr[j];
                corr_ptr[gi] = c * (is_diag - 1) + is_diag;
            }
        });
    });

    return finalize_corr_event;
}

#define INSTANTIATE(F)                                                      \
    template ONEDAL_EXPORT sycl::event correlation<F>(sycl::queue&,         \
                                                      const table&,         \
                                                      const ndview<F, 1>&,  \
                                                      ndview<F, 2>&,        \
                                                      ndview<F, 1>&,        \
                                                      ndview<F, 1>&,        \
                                                      ndview<F, 1>&,        \
                                                      const event_vector&);

INSTANTIATE(float)
INSTANTIATE(double)

} // namespace oneapi::dal::backend::primitives
