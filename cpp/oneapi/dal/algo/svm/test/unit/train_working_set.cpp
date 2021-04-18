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

#include "oneapi/dal/algo/svm/backend/gpu/train_working_set_dpc.hpp"

namespace oneapi::dal::svm::backend::test {

namespace pr = dal::backend::primitives;
namespace te = dal::test::engine;
namespace la = te::linalg;
namespace de = dal::detail;

template <typename TestType>
class train_working_set_test : public te::policy_fixture {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Index = std::tuple_element_t<1, TestType>;

// bool not_available_on_device() {
//         constexpr bool is_smo = std::is_same_v<Method, svm::method::smo>;
//         return this->get_policy().is_gpu() && is_smo;
//     }

};

using train_working_set_types = COMBINE_TYPES((float, double), (std::uint32_t));

TEMPLATE_LIST_TEST_M(train_working_set_test, "basic run", "[working_set]", train_working_set_types) {
    using float_t = std::tuple_element_t<0, TestType>;
    // using integer_t = std::tuple_element_t<1, TestType>;
    auto& q = this->get_queue();
    // auto q = sycl::queue{sycl::device{sycl::gpu_selector{}}};

    constexpr std::int64_t n_vectors = 10;

    std::vector<float_t> f = { 1, 3, 10, 4, 2, 8, 6, 5, 9, 7 };
    std::vector<float_t> y = { -1, -1, -1, -1, -1, 1, 1, 1, 1, 1 };
    std::vector<float_t> alpha = { 0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5 };

    const pr::ndarray<float_t, 1> f_ndarray = pr::ndarray<float_t, 1>::wrap(f.data(), n_vectors);
    const pr::ndarray<float_t, 1> y_ndarray = pr::ndarray<float_t, 1>::wrap(y.data(), n_vectors);
    const pr::ndarray<float_t, 1> alpha_ndarray = pr::ndarray<float_t, 1>::wrap(alpha.data(), n_vectors);

    float_t C = 1.5;

    auto ws = working_set<float_t>(q);
    ws.init(n_vectors);
    ws.select_ws(y_ndarray, alpha_ndarray, f_ndarray, C);
    REQUIRE(ws.get_size() == 4);
    
}

} // oneapi::dal::svm::backend::test
