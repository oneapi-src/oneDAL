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

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/io.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/backend/primitives/select/select.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class select_flagged_test : public te::policy_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Flag = std::tuple_element_t<1, TestType>;
    using Integer = std::uint32_t;

    auto allocate_arrays(Integer elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Float, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto mask = ndarray<Flag, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        return std::make_tuple(val, mask);
    }

    void fill_uniform(ndarray<Float, 1>& val,
                      ndarray<Flag, 1>& mask,
                      Float a,
                      Float b,
                      std::int64_t seed = 777) {
        Integer elem_count = de::integral_cast<Integer>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_real_distribution<Float> distr(a, b);

        Float pivot = distr(rng);
        // move generation to device when rng is available there
        Float* val_ptr = detail::host_allocator<Float>().allocate(val.get_count());
        Flag* mask_ptr = detail::host_allocator<Flag>().allocate(val.get_count());

        for (Integer el = 0; el < elem_count; el++) {
            val_ptr[el] = distr(rng);
            mask_ptr[el] = val_ptr[el] < pivot ? 1 : 0;
        }
        val.assign(this->get_queue(), val_ptr, val.get_count()).wait_and_throw();
        mask.assign(this->get_queue(), mask_ptr, val.get_count()).wait_and_throw();

        detail::host_allocator<Float>().deallocate(val_ptr, val.get_count());
        detail::host_allocator<Flag>().deallocate(mask_ptr, val.get_count());
    }

    auto create_reference_on_host(const ndarray<Float, 1>& in, const ndarray<Flag, 1>& mask) {
        const auto in_host = in.to_host(this->get_queue());
        const auto in_host_ptr = in_host.get_data();
        const auto mask_host = mask.to_host(this->get_queue());
        const auto mask_host_ptr = mask_host.get_data();

        Float* ref_ptr = detail::host_allocator<Float>().allocate(in.get_count());

        std::int64_t total_sum = 0;
        for (std::int64_t el = 0; el < in.get_count(); el++) {
            if (mask_host_ptr[el]) {
                ref_ptr[total_sum] = in_host_ptr[el];
                total_sum++;
            }
        }

        auto ref = ndarray<Float, 1>::wrap(ref_ptr, { in.get_count() });

        return std::make_tuple(ref, total_sum);
    }

    void check_select(ndarray<Float, 1>& in, ndarray<Flag, 1>& mask) {
        auto& q = this->get_queue();
        INFO("create reference");
        auto [ref, total_sum_ref] = create_reference_on_host(in, mask);
        INFO("run select");
        auto out = ndarray<Float, 1>::empty(q, { in.get_count() }, sycl::usm::alloc::device);
        std::int64_t total_sum_res;

        auto event =
            selector<Float, Flag>{ this->get_queue() }.select_flagged(mask, in, out, total_sum_res);
        event.wait_and_throw();

        REQUIRE(total_sum_res == total_sum_ref);

        // implement
        check_results(out, ref, total_sum_ref);
    }

    void check_results(const ndarray<Float, 1>& out,
                       const ndarray<Float, 1>& ref_host,
                       std::int64_t total_sum_ref) {
        const auto out_host = out.to_host(this->get_queue());
        const Float* out_ptr = out_host.get_data();
        const Float* ref_ptr = ref_host.get_data();

        for (std::int64_t el = 0; el < total_sum_ref; el++) {
            REQUIRE(out_ptr[el] == ref_ptr[el]);
        }
    }
};

template <typename TestType>
class select_flagged_index_test : public te::policy_fixture {
public:
    using Data = std::tuple_element_t<0, TestType>;
    using Flag = std::tuple_element_t<1, TestType>;
    using Integer = std::uint32_t;

    static_assert(std::numeric_limits<Data>::is_integer);

    auto allocate_arrays(Integer elem_count) {
        auto& q = this->get_queue();
        auto val = ndarray<Data, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);
        auto mask = ndarray<Flag, 1>::empty(q, { elem_count }, sycl::usm::alloc::device);

        return std::make_tuple(val, mask);
    }

    void fill_uniform(ndarray<Data, 1>& val, ndarray<Flag, 1>& mask, std::int64_t seed = 777) {
        Integer elem_count = de::integral_cast<Integer>(val.get_count());
        std::mt19937 rng(seed);
        std::uniform_int_distribution<Data> distr(0, val.get_count() - 1);

        // move generation to device when rng is available there
        Data* val_ptr = detail::host_allocator<Data>().allocate(val.get_count());
        Flag* mask_ptr = detail::host_allocator<Flag>().allocate(val.get_count());
        for (Integer el = 0; el < elem_count; el++) {
            val_ptr[el] = el;
            mask_ptr[el] = 0;
        }

        for (Integer el = 0; el < elem_count; el++) {
            Integer ind = distr(rng);
            mask_ptr[ind] = 1;
            Integer swap_ind = distr(rng);
            std::swap(val_ptr[el], val_ptr[swap_ind]);
        }
        val.assign(this->get_queue(), val_ptr, val.get_count()).wait_and_throw();
        mask.assign(this->get_queue(), mask_ptr, val.get_count()).wait_and_throw();

        detail::host_allocator<Data>().deallocate(val_ptr, val.get_count());
        detail::host_allocator<Flag>().deallocate(mask_ptr, val.get_count());
    }

    auto create_reference_on_host(const ndarray<Data, 1>& in, const ndarray<Flag, 1>& mask) {
        const auto in_host = in.to_host(this->get_queue());
        const auto in_host_ptr = in_host.get_data();
        const auto mask_host = mask.to_host(this->get_queue());
        const auto mask_host_ptr = mask_host.get_data();

        Data* ref_ptr = detail::host_allocator<Data>().allocate(in.get_count());

        std::int64_t total_sum = 0;
        for (std::int64_t el = 0; el < in.get_count(); el++) {
            if (mask_host_ptr[in_host_ptr[el]]) {
                ref_ptr[total_sum] = in_host_ptr[el];
                total_sum++;
            }
        }

        auto ref = ndarray<Data, 1>::wrap(ref_ptr, { in.get_count() });

        return std::make_tuple(ref, total_sum);
    }

    void check_select(ndarray<Data, 1>& in, ndarray<Flag, 1>& mask) {
        auto& q = this->get_queue();
        INFO("create reference");
        auto [ref, total_sum_ref] = create_reference_on_host(in, mask);
        INFO("run select");
        auto out = ndarray<Data, 1>::empty(q, { in.get_count() }, sycl::usm::alloc::device);
        std::int64_t total_sum_res;

        auto event = selector<Data, Flag>{ this->get_queue() }.select_flagged_index(mask,
                                                                                    in,
                                                                                    out,
                                                                                    total_sum_res);
        event.wait_and_throw();

        REQUIRE(total_sum_res == total_sum_ref);

        // implement
        check_results(out, ref, total_sum_ref);
    }

    void check_results(const ndarray<Data, 1>& out,
                       const ndarray<Data, 1>& ref_host,
                       std::int64_t total_sum_ref) {
        const auto out_host = out.to_host(this->get_queue());
        const Data* out_ptr = out_host.get_data();
        const Data* ref_ptr = ref_host.get_data();

        for (std::int64_t el = 0; el < total_sum_ref; el++) {
            REQUIRE(out_ptr[el] == ref_ptr[el]);
        }
    }
};

using select_flagged_types = COMBINE_TYPES((float, double), (std::uint8_t, std::uint32_t));
using select_flagged_index_types = COMBINE_TYPES((std::int32_t, std::uint32_t),
                                                 (std::uint8_t, std::uint32_t));

TEMPLATE_LIST_TEST_M(select_flagged_test,
                     "select flagged",
                     "[select flagged]",
                     select_flagged_types) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t elem_count = GENERATE_COPY(2, 15, 16000);

    auto [val, mask] = this->allocate_arrays(elem_count);
    this->fill_uniform(val, mask, -25., 25.);

    this->check_select(val, mask);
}

TEMPLATE_LIST_TEST_M(select_flagged_index_test,
                     "select flagged index",
                     "[select flagged]",
                     select_flagged_index_types) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t elem_count = GENERATE_COPY(2, 15, 16000);

    auto [val, mask] = this->allocate_arrays(elem_count);
    this->fill_uniform(val, mask);

    this->check_select(val, mask);
}
} // namespace oneapi::dal::backend::primitives::test
