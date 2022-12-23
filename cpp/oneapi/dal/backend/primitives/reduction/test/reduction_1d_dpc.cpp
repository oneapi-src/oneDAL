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

#include "oneapi/dal/backend/primitives/reduction/reduction_1d.hpp"

namespace oneapi::dal::backend::primitives::test {

namespace te = dal::test::engine;
namespace pr = oneapi::dal::backend::primitives;

using std::cout;
using std::endl;

using reduction_types = std::tuple<std::tuple<float, sum<float>, square<float>>,
                                   std::tuple<double, sum<double>, square<double>>>;



template <typename Param>
class reduction_test_random_1d : public te::float_algo_fixture<std::tuple_element_t<0, Param>> {
public:
    using float_t = std::tuple_element_t<0, Param>;
    using binary_t = std::tuple_element_t<1, Param>;
    using unary_t = std::tuple_element_t<2, Param>;

    void generate() {
        n_ = GENERATE(17, 999, 1, 5, 1001);
        CAPTURE(n_);
        generate_input();
    }

    bool is_initialized() const {
        return n_ > 0;
    }

    void check_if_initialized() {
        if (!is_initialized()) {
            throw std::runtime_error{ "reduce test is not initialized" };
        }
    }

    float_t groundtruth() const {
        // ONEDAL_ASSERT(!input_table_.empty());
        float_t res = float_t(binary_.init_value);
        // const auto* const inp_ptr = input.get_data();
        // row_accessor<const float_t>{ input_table_ }.pull({ j, j + 1 });
        auto inp = row_accessor<const float_t>{this->input_table_}.pull({0, 1});
        for (std::int64_t i = 0; i < inp.get_count(); ++i) {
            res = binary_(res, unary_(inp[i]));
        }
        return res;
    }
    /*
    sycl::event random_fill_array(ndview<float_t, 1>& input, const event_vector& deps = {}) const {

        const uint32_t seed = 2007;
        auto inp_ptr = input.get_mutable_data();
        return this->get_queue().submit([&](sycl::handler& cgh){
            cgh.depends_on(deps);
            cgh.parallel_for(make_range_1d(input.get_count()), 
            [=](sycl::id<1> idx) {

                std::uint64_t offset = idx;

                // Create minstd_rand engine
                oneapi::dpl::minstd_rand engine(seed, offset);

                // Create float uniform_real_distribution distribution
                oneapi::dpl::uniform_real_distribution<float_t> distr;

                // Generate float random number
                inp_ptr[idx] = distr(engine);

            });
        });
    } 
    */

    void generate_input() {
        const auto train_dataframe =
            GENERATE_DATAFRAME(te::dataframe_builder{ 1, n_ }.fill_uniform(-0.2, 0.5));
        this->input_table_ = train_dataframe.get_table(this->get_homogen_table_id());
    }


    void test_1d_reduce(const float_t tol = 1.e-3) {
        auto input_array = row_accessor<const float_t>{ input_table_ }.pull(this->get_queue());
        auto input = ndview<float_t, 1>::wrap(input_array.get_data(), { n_ });

        float_t out = reduce_1d(this->get_queue(), input, binary_t{}, unary_t{}, { });

        float_t ans = groundtruth();

        if (out - ans < -tol || out - ans > tol) {
            CAPTURE(out, ans, out - ans, tol);
            FAIL();
        }
        SUCCEED();
    }

    /*
    void test_1d_reduce(const float_t tol = 1.e-3) {

        // auto inp = ndarray<float_t, 1>::empty(this->get_queue(), { n_ });

        // auto inp_e = random_fill_array(inp);

        auto [inp, inp_e] = ndarray<float_t, 1>::full(this->get_queue(), n_, float_t(2), sycl::usm::alloc::device);

        ndview<float_t, 2, ndorder::c>::wrap(input_array.get_data(), { height_, width_ });

        float_t out = reduce_1d(this->get_queue(), inp, binary_t{}, unary_t{}, {inp_e});

        inp_e.wait_and_throw();

        float_t ans = groundtruth(inp.to_host(this->get_queue()));

        if (out - ans < -tol || out - ans > tol) {
            CAPTURE(out, ans, out - ans, tol);
            FAIL();
        }
        SUCCEED();
    }
    */

private:
    table input_table_;
    const binary_t binary_{};
    const unary_t unary_{};
    std::int64_t n_;
};

TEMPLATE_LIST_TEST_M(reduction_test_random_1d,
                     "Array filled with 2",
                     "[reduction][1d][small]",
                     reduction_types) {
    SKIP_IF(this->not_float64_friendly());
    this->generate();
    this->test_1d_reduce();
}

} // oneapi::dal::backend::primitives::test