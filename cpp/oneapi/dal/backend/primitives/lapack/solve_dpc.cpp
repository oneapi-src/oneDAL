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

#include "oneapi/dal/backend/primitives/lapack.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"

namespace oneapi::dal::backend::primitives {

template <bool beta, typename Float, ndorder xlayout, ndorder ylayout>
inline sycl::event beta_copy_transform(sycl::queue& queue,
                                       const ndview<Float, 2, xlayout>& src,
                                       ndview<Float, 2, ylayout>& dst,
                                       const event_vector& dependencies) {
    ONEDAL_ASSERT(src.has_data());
    const auto shape = dst.get_shape();
    ONEDAL_ASSERT(dst.has_mutable_data());

    ONEDAL_ASSERT(shape.at(0) == src.get_dimension(0));
    ONEDAL_ASSERT(shape.at(1) == src.get_dimension(1) + !beta);

    return queue.submit([&](sycl::handler& h) {
        h.depends_on(dependencies);

        auto src_ndx = make_ndindexer(src);
        auto dst_ndx = make_ndindexer(dst);

        const auto w = shape.at(1);
        const auto range = shape.to_range();
        h.parallel_for(range, [=](sycl::id<2> idx) {
            const auto r = idx[0];
            const auto c = idx[1];

            if (c == 0) {
                if constexpr (beta) {
                    dst_ndx.at(r, c) = src_ndx.at(r, w - 1);
                }
                else {
                    dst_ndx.at(r, c) = Float(0);
                }
            }
            else {
                dst_ndx.at(r, c) = src_ndx.at(r, c - 1);
            }
        });
    });
}

template <mkl::uplo uplo, bool beta, typename Float, ndorder xlayout, ndorder ylayout>
sycl::event solve_system(sycl::queue& queue,
                         const ndview<Float, 2, xlayout>& xtx,
                         const ndview<Float, 2, ylayout>& xty,
                         ndview<Float, 2, ndorder::c>& final_xtx,
                         ndview<Float, 2, ndorder::c>& final_xty,
                         const event_vector& dependencies) {
    constexpr auto alloc = sycl::usm::alloc::device;

    auto [nxty, xty_event] = copy<ndorder::c, Float, ylayout, alloc>(queue, xty, dependencies);
    auto [nxtx, xtx_event] = copy<ndorder::c, Float, xlayout, alloc>(queue, xtx, dependencies);

    opt_array<Float> dummy{};
    auto potrf_event = potrf_factorization<uplo>(queue, nxtx, dummy, { xtx_event });
    auto potrs_event = potrs_solution<uplo>(queue, nxtx, nxty, dummy, { potrf_event, xty_event });

    return beta_copy_transform<beta>(queue, nxty, final_xty, { potrs_event });
}

#define INSTANTIATE(U, B, F, XL, YL)                                 \
    template sycl::event solve_system<U, B>(sycl::queue&,            \
                                            const ndview<F, 2, XL>&, \
                                            const ndview<F, 2, YL>&, \
                                            ndview<F, 2>&,           \
                                            ndview<F, 2>&,           \
                                            const event_vector&);

#define INSTANTIATE_YL(U, B, F, XL)      \
    INSTANTIATE(U, B, F, XL, ndorder::f) \
    INSTANTIATE(U, B, F, XL, ndorder::c)

#define INSTANTIATE_XL(U, B, F)         \
    INSTANTIATE_YL(U, B, F, ndorder::f) \
    INSTANTIATE_YL(U, B, F, ndorder::c)

#define INSTANTIATE_F(U, B)     \
    INSTANTIATE_XL(U, B, float) \
    INSTANTIATE_XL(U, B, double)

#define INSTANTIATE_B(U)   \
    INSTANTIATE_F(U, true) \
    INSTANTIATE_F(U, false)

INSTANTIATE_B(mkl::uplo::upper)
INSTANTIATE_B(mkl::uplo::lower)

} // namespace oneapi::dal::backend::primitives
