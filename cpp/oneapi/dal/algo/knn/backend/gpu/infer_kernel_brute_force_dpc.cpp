/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#include "oneapi/dal/backend/interop/common_dpc.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/table_conversion.hpp"

#include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
#include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
#include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
#include "oneapi/dal/algo/knn/backend/model_impl.hpp"

#include "oneapi/dal/backend/primitives/common.hpp"
#include "oneapi/dal/backend/primitives/ndarray.hpp"
#include "oneapi/dal/backend/primitives/search.hpp"
#include "oneapi/dal/backend/primitives/selection.hpp"
#include "oneapi/dal/backend/primitives/voting.hpp"
#include "oneapi/dal/backend/primitives/utils.hpp"

#include "oneapi/dal/table/row_accessor.hpp"

#include "oneapi/dal/detail/common.hpp"

namespace oneapi::dal::knn::backend {

using idx_t = std::int32_t;

using dal::backend::context_gpu;

template <typename Task>
using descriptor_t = detail::descriptor_base<Task>;

namespace bk = ::oneapi::dal::backend;
namespace pr = ::oneapi::dal::backend::primitives;

template <typename Float>
sycl::event sqrt(sycl::queue& q, array<Float>& data, const bk::event_vector& deps = {}) {
    ONEDAL_ASSERT(data.has_mutable_data());
    const auto length = data.get_count();
    const auto range = bk::make_range_1d(length);
    auto* const data_ptr = data.get_mutable_data();
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(range, [=](sycl::id<1> idx) {
            data_ptr[idx] = sycl::sqrt(data_ptr[idx]);
        });
    });
}

template <typename Float>
class knn_callback {
    using voting_t = std::unique_ptr<pr::uniform_voting<idx_t>>;

public:
    knn_callback(sycl::queue& q,
                 result_option_id results,
                 std::int64_t query_block,
                 std::int64_t query_length,
                 std::int64_t k_neighbors)
            : queue_(q),
              result_options_(results),
              query_block_(query_block),
              query_length_(query_length),
              k_neighbors_(k_neighbors),
              voting_(pr::make_uniform_voting(q, query_block, k_neighbors)) {
        if (result_options_.test(result_options::responses)) {
            this->temp_resp_ = pr::ndarray<idx_t, 2>::empty(q,
                                                            { query_block, k_neighbors },
                                                            sycl::usm::alloc::device);
        }
    }

    auto& set_inp_responses(const pr::ndview<idx_t, 1>& inp_responses) {
        if (result_options_.test(result_options::responses)) {
            this->inp_responses_ = inp_responses;
        }
        return *this;
    }

    auto& set_responses(const array<idx_t>& responses) {
        if (result_options_.test(result_options::responses)) {
            ONEDAL_ASSERT(responses.get_count() == query_length_);
            this->responses_ = pr::ndarray<idx_t, 1>::wrap_mutable(responses, query_length_);
        }
        return *this;
    }

    auto& set_indices(const array<idx_t>& indices) {
        if (result_options_.test(result_options::indices)) {
            ONEDAL_ASSERT(indices.get_count() ==
                          dal::detail::check_mul_overflow(query_length_, k_neighbors_));
            this->indices_ =
                pr::ndarray<idx_t, 2>::wrap_mutable(indices, { query_length_, k_neighbors_ });
        }
        return *this;
    }

    auto& set_distances(array<Float>& distances) {
        if (result_options_.test(result_options::distances)) {
            ONEDAL_ASSERT(distances.get_count() ==
                          dal::detail::check_mul_overflow(query_length_, k_neighbors_));
            this->distances_ =
                pr::ndarray<Float, 2>::wrap_mutable(distances, { query_length_, k_neighbors_ });
        }
        return *this;
    }

    auto get_blocking() const {
        return bk::uniform_blocking(query_length_, query_block_);
    }

    sycl::event operator()(std::int64_t qb_id,
                           const pr::ndview<idx_t, 2>& inp_indices,
                           const pr::ndview<Float, 2>& inp_distances,
                           const bk::event_vector& deps = {}) {
        sycl::event copy_indices, copy_distances, comp_responses;
        const auto blocking = this->get_blocking();

        const auto from = blocking.get_block_start_index(qb_id);
        const auto to = blocking.get_block_end_index(qb_id);

        if (result_options_.test(result_options::indices)) {
            auto out_block = indices_.get_row_slice(from, to);
            copy_indices = copy_by_value(queue_, out_block, inp_indices, deps);
        }

        if (result_options_.test(result_options::distances)) {
            auto out_block = distances_.get_row_slice(from, to);
            copy_distances = copy_by_value(queue_, out_block, inp_distances, deps);
        }

        if (result_options_.test(result_options::responses)) {
            using namespace bk;
            auto out_block = responses_.get_slice(from, to);
            const auto ndeps = deps + copy_indices + copy_distances;
            auto temp_resp = temp_resp_.get_row_slice(from, to);
            auto s_event = select_indexed(queue_, inp_indices, inp_responses_, temp_resp, ndeps);
            comp_responses = voting_->operator()(temp_resp, out_block, { s_event });
        }

        sycl::event::wait_and_throw({ copy_indices, copy_distances, comp_responses });
        return sycl::event();
    }

private:
    sycl::queue& queue_;
    const result_option_id result_options_;
    const std::int64_t query_block_, query_length_, k_neighbors_;
    pr::ndview<idx_t, 1> inp_responses_;
    pr::ndarray<idx_t, 2> temp_resp_;
    pr::ndarray<idx_t, 1> responses_;
    pr::ndarray<Float, 2> distances_;
    pr::ndarray<idx_t, 2> indices_;
    voting_t voting_;
};

template <typename Float, typename Task>
static infer_result<Task> call_daal_kernel(const context_gpu& ctx,
                                           const descriptor_t<Task>& desc,
                                           const table& infer,
                                           const model<Task>& m) {
    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ dal::detail::error_messages::unknown_distance_type() };
    }
    else if (distance_impl->get_daal_distance_type() != detail::v1::daal_distance_t::minkowski) {
        throw internal_error{ dal::detail::error_messages::distance_is_not_supported_for_gpu() };
    }

    auto& queue = ctx.get_queue();
    bk::interop::execution_context_guard guard(queue);

    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);
    const auto train = trained_model->get_data();
    const auto resps = trained_model->get_responses();

    const std::int64_t infer_row_count = infer.get_row_count();
    const std::int64_t feature_count = train.get_column_count();
    const std::int64_t neighbor_count = desc.get_neighbor_count();
    ONEDAL_ASSERT(train.get_column_count() == infer.get_column_count());

    auto arr_responses = array<idx_t>{};
    if (desc.get_result_options().test(result_options::responses)) {
        arr_responses = array<idx_t>::empty(queue, infer_row_count);
    }
    auto arr_distances = array<Float>{};
    if (desc.get_result_options().test(result_options::distances)) {
        const auto length = dal::detail::check_mul_overflow(infer_row_count, neighbor_count);
        arr_distances = array<Float>::empty(queue, length);
    }
    auto arr_indices = array<idx_t>{};
    if (desc.get_result_options().test(result_options::indices)) {
        const auto length = dal::detail::check_mul_overflow(infer_row_count, neighbor_count);
        arr_indices = array<idx_t>::empty(queue, length);
    }

    auto train_data = pr::table2ndarray<Float>(queue, train, sycl::usm::alloc::device);
    auto query_data = pr::table2ndarray<Float>(queue, infer, sycl::usm::alloc::device);
    auto resps_data = desc.get_result_options().test(result_options::responses)
                          ? pr::table2ndarray_1d<idx_t>(queue, resps, sycl::usm::alloc::device)
                          : pr::ndarray<idx_t, 1>{};

    const std::int64_t infer_block = pr::propose_query_block<Float>(queue, feature_count);
    const std::int64_t train_block = pr::propose_train_block<Float>(queue, feature_count);

    knn_callback<Float> callback(queue,
                                 desc.get_result_options(),
                                 infer_block,
                                 infer_row_count,
                                 neighbor_count);

    callback.set_inp_responses(resps_data);
    callback.set_responses(arr_responses);
    callback.set_distances(arr_distances);
    callback.set_indices(arr_indices);

    if (distance_impl->get_degree() == 2.0) {
        using dst_t = pr::squared_l2_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t>;

        const dst_t dist{ queue };
        const search_t search{ queue, train_data, train_block, dist };
        auto last_event = search(query_data, callback, infer_block, neighbor_count);
        if (desc.get_result_options().test(result_options::distances)) {
            last_event = sqrt<Float>(queue, arr_distances, { last_event });
        }
        last_event.wait_and_throw();
    }
    else {
        using met_t = pr::lp_metric<Float>;
        using dst_t = pr::lp_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t>;

        const dst_t dist{ queue, met_t(distance_impl->get_degree()) };
        const search_t search{ queue, train_data, train_block, dist };
        search(query_data, callback, infer_block, neighbor_count).wait_and_throw();
    }

    auto result = infer_result<Task>{}.set_result_options(desc.get_result_options());

    if (desc.get_result_options().test(result_options::responses)) {
        if constexpr (std::is_same_v<Task, task::classification>) {
            result = result.set_responses(homogen_table::wrap(arr_responses, infer_row_count, 1));
        }
    }

    if (desc.get_result_options().test(result_options::indices)) {
        result =
            result.set_indices(homogen_table::wrap(arr_indices, infer_row_count, neighbor_count));
    }

    if (desc.get_result_options().test(result_options::distances)) {
        result = result.set_distances(
            homogen_table::wrap(arr_distances, infer_row_count, neighbor_count));
    }

    return result;
}

template <typename Float, typename Task>
static infer_result<Task> infer(const context_gpu& ctx,
                                const descriptor_t<Task>& desc,
                                const infer_input<Task>& input) {
    return call_daal_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
}

template <typename Float, typename Task>
struct infer_kernel_gpu<Float, method::brute_force, Task> {
    infer_result<Task> operator()(const context_gpu& ctx,
                                  const descriptor_t<Task>& desc,
                                  const infer_input<Task>& input) const {
        return infer<Float, Task>(ctx, desc, input);
    }
};

template struct infer_kernel_gpu<float, method::brute_force, task::classification>;
template struct infer_kernel_gpu<double, method::brute_force, task::classification>;
template struct infer_kernel_gpu<float, method::brute_force, task::search>;
template struct infer_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
