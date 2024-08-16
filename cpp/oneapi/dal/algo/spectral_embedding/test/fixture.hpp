/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "oneapi/dal/algo/spectral_embedding/compute.hpp"

#include "oneapi/dal/test/engine/common.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"
#include "oneapi/dal/test/engine/dataframe.hpp"
#include "oneapi/dal/test/engine/math.hpp"
#include "oneapi/dal/detail/debug.hpp"
#include <random>

namespace oneapi::dal::spectral_embedding::test {

namespace te = dal::test::engine;
namespace de = dal::detail;
namespace sp_emb = oneapi::dal::spectral_embedding;

using dal::detail::operator<<;

template <typename TestType, typename Derived>
class spectral_embedding_test : public te::crtp_algo_fixture<TestType, Derived> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;
    using input_t = sp_emb::compute_input<>;
    using result_t = sp_emb::compute_result<>;
    using descriptor_t = sp_emb::descriptor<Float, Method>;

    auto get_descriptor(sp_emb::result_option_id compute_mode) const {
        return descriptor_t()
            .set_embedding_dim(100)
            .set_num_neighbors(1000)
            .set_result_options(compute_mode);
    }

    void gen_input() {
        std::mt19937 rnd(2007 + n_ + p_ + n_ * p_);
        const te::dataframe data_df =
            GENERATE_DATAFRAME(te::dataframe_builder{ n_, p_ }.fill_normal(-0.5, 0.5, 7777));
        data_ = data_df.get_table(this->get_policy(), this->get_homogen_table_id());
    }


    void test_default() {
        std::cout << "Input" << std::endl;
        // std::cout << data_ << std::endl;
        auto desc =
            get_descriptor(sp_emb::result_options::embedding);
        //desc.set_embedding_dim(5);
        //desc.set_num_neighbors(4);
        INFO("run compute");
        auto compute_result = this->compute(desc, data_);
        check_compute_result(compute_result);
        std::cout << "Output" << std::endl;
        // std::cout << compute_result.get_embedding() << std::endl;
    }

    void check_compute_result(const spectral_embedding::compute_result<>& result) {
        array<Float> data_arr = row_accessor<const Float>(data_).pull({ 0, -1 });
    }

protected:
    std::int64_t n_;
    std::int64_t p_;
    table data_;
};

using spectral_embedding_types = COMBINE_TYPES((float, double), (sp_emb::method::dense_batch));

} // namespace oneapi::dal::spectral_embedding::test
