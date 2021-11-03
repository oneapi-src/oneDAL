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

#pragma once

#include "oneapi/dal/backend/primitives/utils.hpp"
#include "oneapi/dal/util/common.hpp"

#ifdef ONEDAL_DATA_PARALLEL

namespace oneapi::dal::covariance::backend {

namespace de = dal::detail;
namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

enum class cov_list : std::uint64_t { mean = 1UL << 0, cov = 1UL << 1, cor = 1UL << 2 };

inline constexpr cov_list operator&(cov_list value_left, cov_list value_right) {
    return bitwise_and(value_left, value_right);
}

inline constexpr cov_list operator|(cov_list value_left, cov_list value_right) {
    return bitwise_or(value_left, value_right);
}

constexpr inline cov_list cov_mode_mean = (cov_list::mean);
constexpr inline cov_list cov_mode_cov = (cov_list::cov);
constexpr inline cov_list cov_mode_cor = (cov_list::cor);
constexpr inline cov_list cov_mode_cov_mean = (cov_list::cov | cov_list::mean);
constexpr inline cov_list cov_mode_cov_cor = (cov_list::cov | cov_list::cor);
constexpr inline cov_list cov_mode_cor_mean = (cov_list::cor | cov_list::mean);
constexpr inline cov_list cov_mode_all = (cov_list::mean | cov_list::cov | cov_list::cor);

template <typename Float, cov_list List>
class local_result {
    using alloc = sycl::usm::alloc;
    using own_t = local_result<Float, List>;

public:
    static own_t empty(sycl::queue& q, std::int64_t count) {
        own_t res;
        res.rsum_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        res.rsum2cent_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        res.rmean_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        if constexpr (check_mask_flag(cov_list::cov | cov_list::cor, List)) {
            res.rvarc_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }
        return res;
    }

    auto& get_sum() const {
        return rsum_;
    }
    auto& get_sum2cent() const {
        return rsum2cent_;
    }
    auto& get_mean() const {
        return rmean_;
    }
    auto& get_varc() const {
        return rvarc_;
    }

private:
    local_result() = default;

    pr::ndarray<Float, 1> rsum_;
    pr::ndarray<Float, 1> rmean_;
    pr::ndarray<Float, 1> rsum2cent_;
    pr::ndarray<Float, 1> rvarc_;
};

template <typename Float, cov_list List>
class local_buffer_list {
    using alloc = sycl::usm::alloc;
    using own_t = local_buffer_list<Float, List>;

public:
    static own_t empty(sycl::queue& q, std::int64_t count) {
        own_t res;
        res.rrow_count_ = pr::ndarray<std::int64_t, 1>::empty(q, { count }, alloc::device);
        res.rsum_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        res.rsum2cent_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        res.rmean_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        if constexpr (check_mask_flag(cov_list::cov | cov_list::cor, List)) {
            res.rvarc_ = pr::ndarray<Float, 1>::empty(q, { count }, alloc::device);
        }

        return res;
    }
    auto& get_rc_list() const {
        return rrow_count_;
    }
    auto& get_sum() const {
        return rsum_;
    }
    auto& get_sum2cent() const {
        return rsum2cent_;
    }
    auto& get_mean() const {
        return rmean_;
    }
    auto& get_varc() const {
        return rvarc_;
    }

private:
    local_buffer_list() = default;

    pr::ndarray<std::int64_t, 1> rrow_count_;
    pr::ndarray<Float, 1> rsum_;
    pr::ndarray<Float, 1> rmean_;
    pr::ndarray<Float, 1> rsum2cent_;
    pr::ndarray<Float, 2> rcov_matrix_;
    pr::ndarray<Float, 2> rcor_matrix_;
    pr::ndarray<Float, 1> rvarc_;
};

} // namespace oneapi::dal::covariance::backend
#endif // ONEDAL_DATA_PARALLEL
