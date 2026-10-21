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

#include <exception>
#include <string>
#include <vector>

#include "oneapi/dal/algo/hdbscan/compute.hpp"
#include "oneapi/dal/algo/hdbscan/test/data.hpp"
#include "oneapi/dal/detail/error_messages.hpp"
#include "oneapi/dal/test/engine/fixtures.hpp"

namespace oneapi::dal::hdbscan::test {

namespace te = dal::test::engine;

template <typename TestType>
class hdbscan_spmd_test : public te::float_algo_fixture<std::tuple_element_t<0, TestType>> {
public:
    using Float = std::tuple_element_t<0, TestType>;
    using Method = std::tuple_element_t<1, TestType>;

    /// Run `compute` under the SPMD policy on `rank_count` ranks of a thread communicator.
    /// Exceptions cannot leave a rank thread, so each rank reports the message it got, or an
    /// empty string if it got a result.
    std::vector<std::string> compute_on_ranks(std::int64_t rank_count) {
#ifdef ONEDAL_DATA_PARALLEL
        te::thread_communicator<spmd::device_memory_access::usm> comm{ this->get_queue(),
                                                                       rank_count };
#else
        te::thread_communicator<spmd::device_memory_access::none> comm{ rank_count };
#endif
        const auto desc = hdbscan::descriptor<Float, Method>(5, 5);
        const auto data = make_table<Float>(two_blob_data, two_blob_row_count, 2);
        return comm.map([&](std::int64_t) -> std::string {
            try {
                te::spmd_compute(this->get_policy(), comm, desc, data);
            }
            catch (const std::exception& e) {
                return e.what();
            }
            return "";
        });
    }
};

using hdbscan_types = COMBINE_TYPES((float, double),
                                    (hdbscan::method::brute_force,
                                     hdbscan::method::kd_tree,
                                     hdbscan::method::ball_tree));

TEMPLATE_LIST_TEST_M(hdbscan_spmd_test,
                     "hdbscan: the spmd policy is rejected on every rank",
                     "[hdbscan][spmd]",
                     hdbscan_types) {
    SKIP_IF(this->not_float64_friendly());

    const std::string expected =
        dal::detail::error_messages::spmd_version_of_algorithm_is_not_implemented();
    const auto messages = this->compute_on_ranks(2);
    REQUIRE(messages.size() == 2);
    for (std::size_t rank = 0; rank < messages.size(); rank++) {
        CAPTURE(rank, messages[rank]);
        REQUIRE(messages[rank] == expected);
    }
}

} // namespace oneapi::dal::hdbscan::test
