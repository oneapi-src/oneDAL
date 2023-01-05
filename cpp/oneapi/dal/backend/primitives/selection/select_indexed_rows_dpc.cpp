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

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/ndindexer.hpp"

#include "oneapi/dal/backend/primitives/selection/select_indexed_rows.hpp"

namespace oneapi::dal::backend::primitives {

template <typename Type, typename Index, ndorder inp_ord, ndorder out_ord>
sycl::event select_indexed_rows(sycl::queue& q,
                                const ndview<Index, 1>& ids,
                                const ndview<Type, 2, inp_ord>& src,
                                ndview<Type, 2, out_ord>& dst,
                                const event_vector& deps) {
    const auto h = dst.get_dimension(0);
    const auto w = dst.get_dimension(1);
    const auto range = make_range_2d(h, w);
    auto out_idxr = make_ndindexer(dst);
    auto inp_idxr = make_ndindexer(src);
    const auto* const ids_ptr = ids.get_data();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<2> idx) {
            const auto id = *(ids_ptr + idx[0]);
            const auto val = inp_idxr.at(id, idx[1]);
            out_idxr.at(idx[0], idx[1]) = val;
        });
    });
}

#define INSTANTIATE_FULL(TYPE, INDEX, INPORD, OUTORD)                                 \
    template sycl::event select_indexed_rows(sycl::queue&,                            \
                                             const ndview<INDEX, 1>&,                 \
                                             const ndview<TYPE, 2, ndorder::INPORD>&, \
                                             ndview<TYPE, 2, ndorder::OUTORD>&,       \
                                             const event_vector&);

#define INSTANTIATE_I(TYPE, INDEX, INPORD)   \
    INSTANTIATE_FULL(TYPE, INDEX, INPORD, c) \
    INSTANTIATE_FULL(TYPE, INDEX, INPORD, f)

#define INSTANTIATE_II(TYPE, INDEX) \
    INSTANTIATE_I(TYPE, INDEX, c)   \
    INSTANTIATE_I(TYPE, INDEX, f)

#define INSTANTIATE_III(TYPE)          \
    INSTANTIATE_II(TYPE, std::int32_t) \
    INSTANTIATE_II(TYPE, std::int64_t)

INSTANTIATE_III(std::int32_t)
INSTANTIATE_III(std::int64_t)
INSTANTIATE_III(double)
INSTANTIATE_III(float)

} // namespace oneapi::dal::backend::primitives
