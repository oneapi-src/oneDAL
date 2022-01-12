/*******************************************************************************
* Copyright 2021-2022 Intel Corporation
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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"

namespace oneapi::dal::backend::primitives {

template<typename Float, ndorder ord, typename Idx>
struct dkeeper {};

template<typename Float, typename Idx>
struct dkeeper<Float, ndorder::c, Idx> {

    struct row_iterator {
        const Float& operator* () const {
            return *(row + col);
        }

        row_iterator& operator++ () {
            ++col;
            return *this;
        }

        bool operator!= (const row_iterator& rhs) const {
            const auto& lhs = *this;
            return lhs.col != rhs.col;
        }

        Idx col = 0;
        const Idx str;
        const Float* const row;
    };

    auto get_first_in_row_iterator(Idx idx) const {
        const Float* const row = ptr + idx * str;
        return row_iterator{ Idx(0), str, row };
    }

    auto get_last_in_row_iterator(Idx idx) const {
        const Float* const row = ptr + idx * str;
        return row_iterator{ width, str, row };
    }

    const Float* const ptr;
    const Idx width;
    const Idx str;
};

template<typename Float, typename Idx>
struct dkeeper<Float, ndorder::f, Idx> {

    struct row_iterator {
        const Float& operator* () const {
            return *(row + col * str);
        }

        row_iterator& operator++ () {
            ++col;
            return *this;
        }

        bool operator!= (const row_iterator& rhs) const {
            const auto& lhs = *this;
            return lhs.col != rhs.col;
        }

        Idx col = 0;
        const Idx str;
        const Float* const row;
    };

    auto get_first_in_row_iterator(Idx idx) const {
        const Float* const row = ptr + idx;
        return row_iterator{ Idx(0), str, row };
    }

    auto get_last_in_row_iterator(Idx idx) const {
        const Float* const row = ptr + idx;
        return row_iterator{ width, str, row };
    }

    const Float* const ptr;
    const Idx width;
    const Idx str;
};

template<typename Float, ndorder order, typename Idx = std::int64_t>
auto make_dkeeper(const ndview<Float, 2, order>& data) {
    return dkeeper<Float, order, Idx>{ data.get_data(),
                                       data.get_dimension(1),
                                       data.get_leading_stride() };
}


template <typename Float, typename Metric>
template <ndorder order1, ndorder order2>
sycl::event distance<Float, Metric>::operator()(const ndview<Float, 2, order1>& inp1,
                                                const ndview<Float, 2, order2>& inp2,
                                                ndview<Float, 2>& out,
                                                const event_vector& deps) const {
    check_inputs(inp1, inp2, out);
    // Getting raw USM pointers
    //const auto* inp1_ptr = inp1.get_data();
    //const auto* inp2_ptr = inp2.get_data();
    auto* out_ptr = out.get_mutable_data();
    // Getting info about dimensions
    const auto dkeeper1 = make_dkeeper(inp1);
    const auto dkeeper2 = make_dkeeper(inp2);
    const auto n_samples1 = inp1.get_dimension(0);
    const auto n_samples2 = inp2.get_dimension(0);
    // Getting info about strides
    const auto out_stride = out.get_leading_stride();
    // Constructing correct range of size m x n
    const auto out_range = make_range_2d(n_samples1, n_samples2);
    // Metric instance
    const auto& metric = this->m_;
    return q_.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(out_range, [=](sycl::id<2> idx) {
            auto inp1_first = dkeeper1.get_first_in_row_iterator(idx[0]);
            auto inp2_first = dkeeper2.get_first_in_row_iterator(idx[1]);
            auto& out_place = *(out_ptr + out_stride * idx[0] + idx[1]);
            auto inp1_last = dkeeper1.get_last_in_row_iterator(idx[0]);
            out_place = metric(inp1_first, inp1_last, inp2_first);
        });
    });
}

#define INSTANTIATE(F, A, B)                                              \
template sycl::event distance<F, lp_metric<F>>::operator()(const ndview<F, 2, A>&,    \
                                               const ndview<F, 2, B>&,    \
                                               ndview<F, 2>&,             \
                                               const event_vector&) const;

#define INSTANTIATE_B(F, A)  \
INSTANTIATE(F, A, ndorder::c)\
INSTANTIATE(F, A, ndorder::f)

#define INSTANTIATE_F(F)                    \
INSTANTIATE_B(F, ndorder::c)                \
INSTANTIATE_B(F, ndorder::f)                \
template class distance<F, lp_metric<F>>;

INSTANTIATE_F(float);
INSTANTIATE_F(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
