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

#include <array>
#include <cmath>
#include <type_traits>

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_misc.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using distance_types = std::tuple<float, double>;

template <typename Float>
class reduction_rm_test_uniform : public te::float_algo_fixture<Float> {
public:
    std::int64_t propose_block_size() {
        ONEDAL_ASSERT(width_ > 0);
        // Here we are focusing on GEMM performance
        const auto k = this->width_;
        const auto wg = this->get_queue()
                            .get_device()
                            .template get_info<sycl::info::device::max_work_group_size>();
        const auto cache = this->get_queue()
                               .get_device()
                               .template get_info<sycl::info::device::global_mem_cache_size>();
        const auto cs = cache / sizeof(Float);
        // GEMM is addressing M * N + N * k + M * k memory in total
        // it should fit to cache size cs. For better WG utilization
        // we should use (m, n) % wg == 0. Thus assuming square
        // block: wg^2 * n^2 + 2 * k * wg * n <= cs.
        const Float det = wg * wg * (cs + k * k);
        const std::int64_t sol = (std::sqrt(det) - wg * k) / Float(wg * wg);
        return std::max<std::int64_t>(1, sol) * wg;
    }

    void generate_smart() {
        width_ = GENERATE(28, 784, 2000, 3072);
        height_ = propose_block_size();
        check_if_initialized();
    }

    void generate() {
        width_ = GENERATE(28, 784, 2000, 3072);
        height_ = GENERATE(256, 1024);
        check_if_initialized();
    }

    bool is_initialized() const {
        return width_ > 0 && height_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    auto input(Float value_to_fill = 0.1) {
        check_if_initialized();
        return ndarray<Float, 2>::full(this->get_queue(),
                                       { height_, width_ },
                                       value_to_fill,
                                       sycl::usm::alloc::device);
    }

    auto output() {
        check_if_initialized();
        return ndarray<Float, 2>::empty(this->get_queue(),
                                        { height_, height_ },
                                        sycl::usm::alloc::device);
    }

    auto fpt_desc() {
        if constexpr (std::is_same_v<Float, float>) {
            return "float";
        }
        if constexpr (std::is_same_v<Float, double>) {
            return "double";
        }
        return "unknown type";
    }

    auto type_desc() {
        return fmt::format("Floating Point Type: {}", fpt_desc());
    }

    auto block_desc() {
        check_if_initialized();
        return fmt::format("RM Block with parameters: "
                           "width_ = {}, height_ = {}",
                           width_,
                           height_);
    }

    auto desc() {
        return fmt::format("{}; {}", block_desc(), type_desc());
    }

    void test_squared_l2_distance() {
        const auto [inp1_array, inp1_event] = input(0.1);
        const auto [inp2_array, inp2_event] = input(10.);
        auto out_array = output();

        sycl::event::wait({ inp1_event, inp2_event });

        auto inp1_norms =
            ndarray<Float, 1>::empty(this->get_queue(), { height_ }, sycl::usm::alloc::device);
        auto inp2_norms =
            ndarray<Float, 1>::empty(this->get_queue(), { height_ }, sycl::usm::alloc::device);

        // BENCHMARK is working with lambdas so the following
        // code is a workaround for binding issues
        const ndview<Float, 2>& inp1_a_view = inp1_array;
        const ndview<Float, 2>& inp2_a_view = inp2_array;
        ndview<Float, 1>& inp1_n_view = inp1_norms;
        ndview<Float, 1>& inp2_n_view = inp2_norms;

        this->get_queue().wait_and_throw();
        BENCHMARK(fmt::format("Squared L2 Norms computation: {}", desc()).c_str()) {
            compute_squared_l2_norms(this->get_queue(), inp1_a_view, inp1_n_view).wait_and_throw();
        };
        compute_squared_l2_norms(this->get_queue(), inp2_a_view, inp2_n_view).wait_and_throw();

        this->get_queue().wait_and_throw();
        BENCHMARK(fmt::format("Scatter L2 Norms computation: {}", desc()).c_str()) {
            scatter_2d(this->get_queue(), inp1_n_view, inp2_n_view, out_array).wait_and_throw();
        };

        this->get_queue().wait_and_throw();
        BENCHMARK(fmt::format("Inner-Product computation: {}", desc()).c_str()) {
            compute_inner_product(this->get_queue(), inp1_a_view, inp2_a_view, out_array)
                .wait_and_throw();
        };

        squared_l2_distance<Float> distance(this->get_queue());

        this->get_queue().wait_and_throw();
        BENCHMARK(fmt::format("Squared L2 Distance computation: {}", desc()).c_str()) {
            distance(inp1_a_view, inp2_a_view, out_array, inp1_n_view, inp2_n_view)
                .wait_and_throw();
        };
    }

private:
    table input_table1_;
    table input_table2_;
    std::int64_t width_;
    std::int64_t height_;
};

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled SqL2 Distance"
                     "with grid blocking",
                     "[distance][sql2][small]",
                     distance_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_squared_l2_distance();
}

TEMPLATE_LIST_TEST_M(reduction_rm_test_uniform,
                     "Uniformly filled SqL2 Distance"
                     "with smart blocking",
                     "[distance][sql2][small]",
                     distance_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate_smart();
    this->test_squared_l2_distance();
}

} // namespace oneapi::dal::backend::primitives::test
