/*******************************************************************************
* Copyright contributors to the oneDAL project
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

#include "oneapi/dal/algo/logistic_regression/test/fixture.hpp"

#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::logistic_regression::test {

namespace te = dal::test::engine;
namespace de = dal::detail;

template <typename TestType>
class log_reg_spmd_test : public log_reg_test<TestType, log_reg_spmd_test<TestType>> {
public:
    using base_t = log_reg_test<TestType, log_reg_spmd_test<TestType>>;

    using float_t = typename base_t::float_t;
    using train_input_t = typename base_t::train_input_t;
    using train_result_t = typename base_t::train_result_t;

    train_result_t merge_train_result_override(const std::vector<train_result_t>& results) {
        return results[0];
    }

    void gen_dimensions(std::int64_t n = -1,
                        std::int64_t p = -1,
                        double train_size_coef = 0.7) override {
        if (n == -1 || p == -1) {
            this->n_ = GENERATE(50, 99);
            this->p_ = GENERATE(3, 10);
        }
        else {
            this->n_ = n;
            this->p_ = p;
        }
        this->train_size_ = (this->n_ * train_size_coef);
        this->test_size_ = this->n_ - this->train_size_;
    }

    template <typename... Args>
    std::vector<train_input_t> split_train_input_override(std::int64_t split_count,
                                                          Args&&... args) {
        const train_input_t input{ std::forward<Args>(args)... };

        const auto split_x =
            te::split_table_by_rows<float_t>(this->get_policy(), input.get_data(), split_count);
        const auto split_y = te::split_table_by_rows<float_t>(this->get_policy(),
                                                              input.get_responses(),
                                                              split_count);

        std::vector<train_input_t> split_input;
        split_input.reserve(split_count);

        for (std::int64_t i = 0; i < split_count; ++i) {
            split_input.emplace_back(split_x.at(i), split_y.at(i));
        }

        return split_input;
    }

    void set_rank_count(std::int64_t rank_count) {
        rank_count_ = rank_count;
    }

    template <typename... Args>
    train_result_t train_override(Args&&... args) {
        return this->train_via_spmd_threads_and_merge(rank_count_, std::forward<Args>(args)...);
    }

private:
    std::int64_t rank_count_;
};

using log_reg_spmd_types = COMBINE_TYPES((float, double),
                                         (logistic_regression::method::dense_batch),
                                         (logistic_regression::task::classification));

} // namespace oneapi::dal::logistic_regression::test
