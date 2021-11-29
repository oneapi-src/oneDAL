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

#define INSTANTIATE_FULL(TYPE, INDEX, INPORD, OUTORD)                         \
    template sycl::event select_indexed_rows<TYPE, INDEX, INPORD, OUTORD>(    \
                                             sycl::queue&,                    \
                                             const ndview<INDEX, 1>&,         \
                                             const ndview<TYPE, 2, INPORD>&,  \
                                             ndview<TYPE, 2, OUTORD>&,        \
                                             const event_vector&);

INSTANTIATE_FULL(std::int32_t, std::int32_t, ndorder::c, ndorder::c);

#define INSTANTATE_INPORD(TYPE, INDEX, OUTORD)        \
    INSTANTIATE_FULL(TYPE, INDEX, ndorder::c, OUTORD);\
    INSTANTIATE_FULL(TYPE, INDEX, ndorder::f, OUTORD);

#define INSTANTATE_OUTORD(TYPE, INDEX)          \
    INSTANTIATE_INPORD(TYPE, INDEX, ndorder::c);\
    INSTANTIATE_INPORD(TYPE, INDEX, ndorder::f);

#define INSTANTIATE_TYPE(INDEX)              \
    INSTANTIATE_OUTORD(float, INDEX);        \
    INSTANTIATE_OUTORD(double, INDEX);       \
    INSTANTIATE_OUTORD(std::int32_t, INDEX); \
    INSTANTIATE_OUTORD(std::int64_t, INDEX);

//INSTANTIATE_TYPE(std::int32_t);
//INSTANTIATE_TYPE(std::int64_t);

} // namespace oneapi::dal::backend::primitives
