/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/dal/detail/profiler.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"
#include "oneapi/dal/backend/primitives/element_wise.hpp"

#include "oneapi/dal/algo/linear_regression/backend/gpu/update_kernel.hpp"

namespace oneapi::dal::linear_regression::backend {

template <bool beta, typename Float>
sycl::event symmetrize(sycl::queue& queue,
                       pr::ndview<Float, 2, pr::ndorder::c>& xtx,
                       const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(symmetrize_kernel, queue);

    const auto ext_f_count = xtx.get_dimension(0);
    ONEDAL_ASSERT(ext_f_count == xtx.get_dimension(1));

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(deps);

        const auto index = pr::make_ndindexer(xtx);
        const auto shape = be::make_range_2d(ext_f_count, ext_f_count);
        h.parallel_for(shape, [=](sycl::id<2> idx) {
            const auto r = idx[0], c = idx[1];
            if (r < c)
                index.at(r, c) = index.at(c, r);
        });
    });
}

template <bool beta, typename Float, pr::ndorder layout>
sycl::event update_xtx(sycl::queue& queue,
                       const pr::ndview<Float, 2, layout>& x,
                       pr::ndview<Float, 2, pr::ndorder::c>& xtx,
                       const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(update_xtx_kernel, queue);

    constexpr Float one = 1;
    constexpr pr::sum<Float> plus;
    constexpr pr::identity<Float> identity;

    using uplo = pr::mkl::uplo;
    ONEDAL_ASSERT(x.has_data());
    ONEDAL_ASSERT(xtx.has_mutable_data());

    const auto f_count = x.get_dimension(1);
    const auto ext_f_count = xtx.get_dimension(1);
    ONEDAL_ASSERT(ext_f_count == (f_count + std::int64_t(beta)));

    be::event_vector new_deps{};
    {
        ONEDAL_PROFILER_TASK(update_xtx_syrk_kernel, queue);
        auto core = xtx.get_col_slice(0, f_count).get_row_slice(0, f_count);
        ONEDAL_ASSERT(core.get_dimension(0) == f_count);
        ONEDAL_ASSERT(core.get_dimension(1) == f_count);

        auto syrk_event = pr::syrk<uplo::upper>(queue, x, core, one, one, deps);

        new_deps.push_back(syrk_event);
    }

    if constexpr (beta) {
        auto means_2d = xtx.get_col_slice(0, f_count).get_row_slice(f_count, ext_f_count);
        ONEDAL_ASSERT(means_2d.get_count() == f_count);
        ONEDAL_ASSERT(means_2d.get_stride(1) == 1);

        auto means = means_2d.template reshape<1, pr::ndorder::c>(f_count);
        ONEDAL_ASSERT(means.get_count() == f_count);

        auto count = xtx.get_col_slice(f_count, ext_f_count).get_row_slice(f_count, ext_f_count);
        ONEDAL_ASSERT(count.get_count() == 1);

        const auto s_count = x.get_dimension(0);

        auto count_event = pr::element_wise(queue, plus, count, Float(s_count), count, deps);
        auto means_event = pr::reduce_by_columns(queue, x, means, plus, identity, deps, false);

        new_deps.push_back(means_event);
        new_deps.push_back(count_event);
    }

    return symmetrize<beta>(queue, xtx, new_deps);
}

template <bool beta, typename Float, pr::ndorder xlayout, pr::ndorder ylayout>
sycl::event update_xty(sycl::queue& queue,
                       const pr::ndview<Float, 2, xlayout>& x,
                       const pr::ndview<Float, 2, ylayout>& y,
                       pr::ndview<Float, 2, pr::ndorder::f>& xty,
                       const be::event_vector& deps) {
    ONEDAL_PROFILER_TASK(update_xty_kernel, queue);

    constexpr Float one = 1;
    constexpr pr::sum<Float> plus;
    constexpr pr::identity<Float> ident;

    ONEDAL_ASSERT(x.has_data());
    ONEDAL_ASSERT(y.has_data());

    const auto r_count = y.get_dimension(1);
    const auto f_count = x.get_dimension(1);
    const auto ext_f_count = xty.get_dimension(1);
    ONEDAL_ASSERT(r_count == xty.get_dimension(0));
    ONEDAL_ASSERT(x.get_dimension(0) == y.get_dimension(0));
    ONEDAL_ASSERT(ext_f_count == (f_count + std::int64_t(beta)));

    ONEDAL_ASSERT(xty.get_dimension(1) == ext_f_count);
    ONEDAL_ASSERT(xty.get_dimension(0) == r_count);

    be::event_vector new_deps{};
    {
        ONEDAL_PROFILER_TASK(update_xty_gemm_kernel, queue);
        auto core = xty.get_col_slice(0, f_count).t();
        ONEDAL_ASSERT(core.get_dimension(0) == f_count);
        ONEDAL_ASSERT(core.get_dimension(1) == r_count);

        auto gemm_event = pr::gemm(queue, x.t(), y, core, one, one, deps);

        new_deps.push_back(gemm_event);
    }

    if constexpr (beta) {
        auto means_2d = xty.get_col_slice(f_count, ext_f_count);
        auto means = means_2d.template reshape<1, pr::ndorder::c>(r_count);

        ONEDAL_ASSERT(means_2d.get_stride(0) == 1);
        ONEDAL_ASSERT(means_2d.get_count() == r_count);

        auto means_event = pr::reduce_by_columns(queue, y, means, plus, ident, deps, false);

        new_deps.push_back(means_event);
    }

    return be::wait_or_pass(new_deps);
}

#define INSTANTIATE(B, F, XL, YL)                                         \
    template sycl::event update_xty<B>(sycl::queue&,                      \
                                       const pr::ndview<F, 2, XL>&,       \
                                       const pr::ndview<F, 2, YL>&,       \
                                       pr::ndview<F, 2, pr::ndorder::f>&, \
                                       const be::event_vector&);

#define INSTANTIATE_YL(B, F, XL)                                          \
    INSTANTIATE(B, F, XL, pr::ndorder::c)                                 \
    INSTANTIATE(B, F, XL, pr::ndorder::f)                                 \
    template sycl::event update_xtx<B>(sycl::queue&,                      \
                                       const pr::ndview<F, 2, XL>&,       \
                                       pr::ndview<F, 2, pr::ndorder::c>&, \
                                       const be::event_vector&);

#define INSTANTIATE_LAYOUT(B, F)         \
    INSTANTIATE_YL(B, F, pr::ndorder::c) \
    INSTANTIATE_YL(B, F, pr::ndorder::f)

#define INSTANTIATE_FLOAT(B)     \
    INSTANTIATE_LAYOUT(B, float) \
    INSTANTIATE_LAYOUT(B, double)

INSTANTIATE_FLOAT(true);
INSTANTIATE_FLOAT(false);

} // namespace oneapi::dal::linear_regression::backend
