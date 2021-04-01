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

#include <type_traits>
#include <tuple>

#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/table/row_accessor.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;

template <typename T>
struct type2str {
    static const char* name;
};

#define INSTANTIATE_TYPE_MAP(T) \
    template <>                 \
    const char* type2str<T>::name = #T;

INSTANTIATE_TYPE_MAP(float);
INSTANTIATE_TYPE_MAP(double)

template <typename TestType>
class pattern_test : public te::policy_fixture {
public:
    using Float = TestType;

    te::table_id get_homogen_table_id() const {
        return te::table_id::homogen<Float>();
    }

    auto allocate_matrices(std::int64_t k, std::int64_t row_count, std::int64_t col_count) {
        auto data = ndarray<Float, 2>::empty(this->get_queue(), { row_count, col_count }, sycl::usm::alloc::device);
        auto selection = ndarray<Float, 2>::empty(this->get_queue(), { row_count, k }, sycl::usm::alloc::device);
        auto indices = ndarray<std::int32_t, 2>::empty(this->get_queue(), { row_count, k }, sycl::usm::alloc::device);
        return std::make_tuple(data, selection, indices);
    }

    void fill_constant(ndarray<Float, 2>& data, Float a) {
        auto count = data.get_count();
        Float* data_ptr = data.get_mutable_data();

         auto event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(count), [=](sycl::item<1> item) {
                std::int32_t ind = item.get_id()[0];
                data_ptr[ind] = ind;
            });
        });
        event.wait_and_throw();
    }
    void run_simple_rw_reduction(std::int64_t row_count, std::int64_t col_count) {
        INFO("benchmark rw_reduction pattern");
        const auto name =
            fmt::format("Reduction: val_type {}, elem_count {}",
                        type2str<Float>::name,
                        row_count * col_count);

        auto data = ndarray<Float, 2>::empty(this->get_queue(), { row_count, col_count }, sycl::usm::alloc::device);
        auto res = ndarray<Float, 1>::empty(this->get_queue(), { row_count }, sycl::usm::alloc::device);
        auto data_ptr = data.get_mutable_data();
        auto res_ptr = res.get_mutable_data();
        auto event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(row_count * col_count), [=](sycl::item<1> item) {
                std::int32_t ind = item.get_id()[0];
                data_ptr[ind] = 1.0;
            });
        });
        event.wait_and_throw();

        sycl::range<2> global(16, row_count);
        sycl::range<2> local(16, 1);
        sycl::nd_range<2> nd_range2d(global, local);

        this->get_queue().wait_and_throw();
        BENCHMARK(name.c_str()) {
            auto event = this->get_queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
                    const auto row_id = item.get_global_id(1);
                    auto sg = item.get_sub_group();
                    const uint32_t sg_id = sg.get_group_id()[0];
                    if (sg_id > 0)
                        return;
                    const uint32_t local_id = sg.get_local_id()[0];
                    const uint32_t local_range = sg.get_local_range()[0];
                    Float sum = 0.0;
                    for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                        auto val = data_ptr[i + row_id * col_count];
                        sum += val;
                    }
                    sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                    if(local_id == 0)
                        res_ptr[row_id] = sum;
                });
            });
            event.wait_and_throw();
        };
    }

    void run_reduction_fused_with_private_simple(std::int64_t row_count, std::int64_t col_count) {
        INFO("benchmark read");
        const auto name =
            fmt::format("Selection (small k): val_type {}, elem_count {}",
                        type2str<Float>::name,
                        row_count * col_count);

        auto data = ndarray<Float, 2>::empty(this->get_queue(), { row_count, col_count }, sycl::usm::alloc::device);
        auto res = ndarray<Float, 1>::empty(this->get_queue(), { row_count }, sycl::usm::alloc::device);
        auto data_ptr = data.get_mutable_data();
        auto res_ptr = res.get_mutable_data();
        auto event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(row_count * col_count), [=](sycl::item<1> item) {
                std::int32_t ind = item.get_id()[0];
                data_ptr[ind] = 1.0;
            });
        });
        event.wait_and_throw();

        sycl::range<2> global(16, row_count);
        sycl::range<2> local(16, 1);
        sycl::nd_range<2> nd_range2d(global, local);

        this->get_queue().wait_and_throw();
        BENCHMARK(name.c_str()) {
            auto event = this->get_queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
                    const auto row_id = item.get_global_id(1);
                    auto sg = item.get_sub_group();
                    const uint32_t sg_id = sg.get_group_id()[0];
                    if (sg_id > 0)
                        return;
                    Float buf[32];
                    int count[32];
                    for(int j = 0; j < 32; j++)
                        buf[j] = 0;
                    const uint32_t local_id = sg.get_local_id()[0];
                    const uint32_t local_range = sg.get_local_range()[0];
                    Float sum = 0.0;
                    for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                        auto val = data_ptr[i + row_id * col_count];
                        for(int j = 1; j < 32; j++) {
                                buf[j] += val;
                                count[j] += val > 1.5 ? 0 : 1;
                        }
                        sum += val;
                    }
                    sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                    if(local_id == 0)
                        res_ptr[row_id] = sum + buf[count[row_id % 32] % 32];
                });
            });
            event.wait_and_throw();
        };
    }
    void run_reduction_fused_with_private_complex(std::int64_t row_count, std::int64_t col_count) {
        INFO("benchmark read");
        const auto name =
            fmt::format("Selection (small k): val_type {}, elem_count {}",
                        type2str<Float>::name,
                        row_count * col_count);

        auto data = ndarray<Float, 2>::empty(this->get_queue(), { row_count, col_count }, sycl::usm::alloc::device);
        auto res = ndarray<Float, 1>::empty(this->get_queue(), { row_count }, sycl::usm::alloc::device);
        auto data_ptr = data.get_mutable_data();
        auto res_ptr = res.get_mutable_data();
        auto event = this->get_queue().submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<1>(row_count * col_count), [=](sycl::item<1> item) {
                std::int32_t ind = item.get_id()[0];
                data_ptr[ind] = 1.0;
            });
        });
        event.wait_and_throw();

        sycl::range<2> global(16, row_count);
        sycl::range<2> local(16, 1);
        sycl::nd_range<2> nd_range2d(global, local);

        this->get_queue().wait_and_throw();
        BENCHMARK(name.c_str()) {
            auto event = this->get_queue().submit([&](sycl::handler& cgh) {
                cgh.parallel_for(nd_range2d, [=](sycl::nd_item<2> item) {
                    const auto row_id = item.get_global_id(1);
                    auto sg = item.get_sub_group();
                    const uint32_t sg_id = sg.get_group_id()[0];
                    if (sg_id > 0)
                        return;
                    Float buf[32];
                    int count[32];
                    for(int j = 0; j < 32; j++)
                        buf[j] = 0;
                    const uint32_t local_id = sg.get_local_id()[0];
                    const uint32_t local_range = sg.get_local_range()[0];
                    Float sum = 0.0;
                    for (std::uint32_t i = local_id; i < col_count; i += local_range) {
                        auto val = data_ptr[i + row_id * col_count];
                        for(int j = 1; j < 32; j++) {
                            buf[j] += val > 0 ? buf[j - 1] : buf[j];
                            count[j] += val > 1.5 ? count[j - 1] : count[j];
                        }
                        sum += val;
                    }
                    sum = reduce(sg, sum, sycl::ONEAPI::plus<Float>());
                    if(local_id == 0)
                        res_ptr[row_id] = sum + buf[count[row_id % 32] % 32];
                });
            });
            event.wait_and_throw();
        };
    }
};

using pattern_types = std::tuple<float, double>;
TEMPLATE_LIST_TEST_M(pattern_test, "benchmark for simple rw reducton", "[patterns][perf]", pattern_types) {
    SKIP_IF(this->get_policy().is_cpu());

    std::int64_t row_count = GENERATE_COPY(1024);
    std::int64_t col_count = GENERATE_COPY(16 * 1024);
    this->run_simple_rw_reduction(row_count, col_count);
}

using selection_types = std::tuple<float, double>;
TEMPLATE_LIST_TEST_M(pattern_test, "benchmark for rw reduction fused with simple private memory manipulations", "[patterns][perf]", pattern_types) {
    SKIP_IF(this->get_policy().is_cpu());
    std::int64_t row_count = GENERATE_COPY(1024);
    std::int64_t col_count = GENERATE_COPY(16 * 1024);
    this->run_reduction_fused_with_private_simple(row_count, col_count);
}

using selection_types = std::tuple<float, double>;
TEMPLATE_LIST_TEST_M(pattern_test, "benchmark for raw selection", "[patterns][perf]", pattern_types) {
    SKIP_IF(this->get_policy().is_cpu());
    std::int64_t row_count = GENERATE_COPY(1024);
    std::int64_t col_count = GENERATE_COPY(16 * 1024);
    this->run_reduction_fused_with_private_complex(row_count, col_count);
}


} // namespace oneapi::dal::backend::primitives::test
