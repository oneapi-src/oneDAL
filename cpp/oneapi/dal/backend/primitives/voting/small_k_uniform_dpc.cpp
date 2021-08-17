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

#include <algorithm>

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"

#include "oneapi/dal/backend/primitives/voting/uniform.hpp"

namespace oneapi::dal::backend::primitives {

template<typename ClassType>
class label_counter {
public:
    label_counter() : label_(-1), count_(0) {}
    label_counter(const ClassType& l, const ClassType& c)
        : label_(l), count_(c) {}
    label_counter(const label_counter& lc)
        : label_(lc.get_label()), count_(lc.get_count()) {}
    label_counter& operator=(const label_counter& lc) = default;

    const ClassType& get_label() const {
        return label_;
    }

    const ClassType& get_count() const {
        return count_;
    }

    label_counter& operator++() {
        count_++;
        return *this;
    }

private:
    ClassType label_, count_;
};

template<typename ClassType = std::int32_t, std::int32_t k_max>
sycl::event small_k_uniform_vote(sycl::queue& q,
                                 const ndview<ClassType, 2>& responses,
                                 ndview<ClassType, 1>& results,
                                 const event_vector& deps) {
    static_assert(k_max > 0);
    ONEDAL_ASSERT(responses.has_data());
    ONEDAL_ASSERT(results.has_mutable_data());
    ONEDAL_ASSERT(k_max >= responses.get_dimension(1));
    ONEDAL_ASSERT(responses.get_dimension(0) == results.get_dimension(0));
    const auto k_current = responses.get_dimension(1);
    if (k_max == k_current) {
        const auto n_samples = responses.get_dimension(0);
        const auto in_stride = responses.get_leading_stride();
        const auto range = make_range_1d(n_samples);
        const auto* const inp_ptr = responses.get_data();
        auto* const out_ptr = results.get_mutable_data();
        return q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            sycl::stream out(2024, 1024, h);
            h.parallel_for(range, [=](sycl::id<1> idx) {
                using cnt_t = label_counter<ClassType>;
                const auto comp_label = [](const cnt_t& l, const cnt_t& r) -> bool {
                    return l.get_label() < r.get_label();
                };
                const auto comp_count = [](const cnt_t& l, const cnt_t& r) -> bool {
                    return l.get_count() < r.get_count();
                };
                cnt_t counters[k_max] = { cnt_t{} };
                std::int32_t label_count = 0;
                const auto* const row = inp_ptr + idx * in_stride;
                for(std::int32_t k = 0; k < k_max; ++k) {
                    const auto& cls = *(row + k);
                    const cnt_t tmp(cls, 1);
                    auto* to = counters + label_count;
                    auto* upper =
                        std::upper_bound(counters, to, tmp, comp_label);
                    auto* prev = upper - 1;
                    if ((upper > counters) && prev->get_label() == cls) {
                        prev->operator++();
                    }
                    else {
                        ++label_count;
                        for(auto* it = to + 1; it > upper; --it) {
                            *it = *(it - 1);
                        }
                        (*upper) = tmp;
                    }
                }
                const auto* max_occ =
                        std::max_element(counters, counters + label_count, comp_count);
                out_ptr[idx] = max_occ->get_label();
            });
        });
    }
    if constexpr (k_max > 1) {
        return small_k_uniform_vote<ClassType, k_max - 1>(q,
                                                          responses,
                                                          results,
                                                          deps);
    }
    ONEDAL_ASSERT(false);
    return sycl::event();
}

template<typename ClassType, std::int32_t kmax>
small_k_uniform_voting<ClassType, kmax>::small_k_uniform_voting(sycl::queue& q)
    : base_t{ q } {}

template<typename ClassType, std::int32_t kmax>
sycl::event small_k_uniform_voting<ClassType, kmax>::operator() (
        const ndview<ClassType, 2>& responses,
        ndview<ClassType, 1>& results,
        const event_vector& deps) {
    return small_k_uniform_vote<ClassType, kmax>(
        this->get_queue(),
        responses,
        results,
        deps);
}

#define INSTANTIATE(CLASS)                  \
template class small_k_uniform_voting<CLASS>;

INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
