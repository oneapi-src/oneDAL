/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dal/algo/pca/test/fixture.hpp"

#include "oneapi/dal/test/engine/tables.hpp"
#include "oneapi/dal/test/engine/io.hpp"

namespace oneapi::dal::pca::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace la = te::linalg;

template <typename TestType>
class pca_params_test : public pca_test<TestType, pca_params_test<TestType>> {
public:
    using base_t = pca_test<TestType, pca_params_test<TestType>>;

    using Float = typename base_t::Float;
    using Method = cov;

    using input_t = typename base_t::input_t;
    using result_t = typename base_t::result_t;
    using descriptor_t = typename base_t::descriptor_t;

    void generate_parameters() {
        this->block_ = GENERATE(140, 512, 1024);
        this->pack_as_struct_ = GENERATE(0, 1);
    }

    auto get_current_parameters() const {
        detail::train_parameters res{};
        res.set_cpu_macro_block(this->block_);
        return res;
    }

    template <typename Desc, typename... Args>
    result_t train_override(Desc&& desc, Args&&... args) {
        REQUIRE(this->block_ > 0);
        const auto params = this->get_current_parameters();
        if (this->pack_as_struct_) {
            return te::float_algo_fixture<Float>::train(std::forward<Desc>(desc),
                                                        params,
                                                        input_t{ std::forward<Args>(args)... });
        }
        else {
            return te::float_algo_fixture<Float>::train(std::forward<Desc>(desc),
                                                        params,
                                                        std::forward<Args>(args)...);
        }
    }

private:
    std::int64_t block_;
    bool pack_as_struct_;
};

TEMPLATE_LIST_TEST_M(pca_params_test, "pca params", "[pca][params]", pca_types) {
    SKIP_IF(this->not_float64_friendly());

    const te::dataframe input =
        GENERATE_DATAFRAME(te::dataframe_builder{ 500, 40 }.fill_uniform(-100, 100, 7777),
                           te::dataframe_builder{ 1000, 20 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 10000, 100 }.fill_uniform(-30, 30, 7777),
                           te::dataframe_builder{ 100000, 20 }.fill_uniform(1, 10, 7777));
    // Homogen floating point type is the same as algorithm's floating point type
    const auto input_data_table_id = this->get_homogen_table_id();

    this->generate_parameters();

    this->general_checks(input, input_data_table_id);
}

} // namespace oneapi::dal::pca::test