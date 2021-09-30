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
#include "oneapi/dal/backend/primitives/regression.hpp"
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

using voting_t = ::oneapi::dal::knn::voting_mode;

namespace de = ::oneapi::dal::detail;
namespace bk = ::oneapi::dal::backend;
namespace pr = ::oneapi::dal::backend::primitives;

using daal_distance_t = daal::algorithms::internal::PairwiseDistanceType;

template <typename Task>
struct task_to_response_map {
    using type = int;
};

template <>
struct task_to_response_map<task::regression> {
    using type = float;
};

template <>
struct task_to_response_map<task::classification> {
    using type = std::int32_t;
};

template <typename Task>
using response_t = typename task_to_response_map<Task>::type;

template <typename T1, typename T2>
sycl::event copy_with_sqrt(sycl::queue& q,
                           const pr::ndview<T2, 2>& src,
                           pr::ndview<T1, 2>& dst,
                           const bk::event_vector& deps = {}) {
    static_assert(de::is_floating_point<T1>());
    static_assert(de::is_floating_point<T2>());
    ONEDAL_ASSERT(src.has_data());
    ONEDAL_ASSERT(dst.has_mutable_data());
    const pr::ndshape<2> dst_shape = dst.get_shape();
    ONEDAL_ASSERT(dst_shape == src.get_shape());
    T1* const dst_ptr = dst.get_mutable_data();
    const T2* const src_ptr = src.get_data();
    const auto dst_stride = dst.get_leading_stride();
    const auto src_stride = src.get_leading_stride();
    const auto cp_range = bk::make_range_2d(dst_shape[0], dst_shape[1]);
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(cp_range, [=](sycl::id<2> idx) {
            T1& dst_ref = *(dst_ptr + idx[0] * dst_stride + idx[1]);
            const T2& val_ref = *(src_ptr + idx[0] * src_stride + idx[1]);
            dst_ref = sycl::sqrt(val_ref);
        });
    });
}

template <typename Float, typename Task>
class knn_callback {
    using dst_t = Float;
    using idx_t = std::int32_t;
    using res_t = response_t<Task>;

    using uniform_voting_t = std::unique_ptr<pr::uniform_voting<res_t>>;
    using distance_voting_t = std::unique_ptr<pr::distance_voting<dst_t>>;
    using uniform_regression_t = std::unique_ptr<pr::uniform_regression<res_t>>;
    using distance_regression_t = std::unique_ptr<pr::distance_regression<dst_t>>;

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
              k_neighbors_(k_neighbors) {
        if (result_options_.test(result_options::responses)) {
            this->temp_resp_ = pr::ndarray<res_t, 2>::empty(q,
                                                            { query_block, k_neighbors },
                                                            sycl::usm::alloc::device);
        }
    }

    auto& set_euclidean_distance(bool is_euclidean_distance) {
        this->compute_sqrt_ = is_euclidean_distance;
        return *this;
    }

    auto& set_inp_responses(const pr::ndview<res_t, 1>& inp_responses) {
        if (result_options_.test(result_options::responses)) {
            this->inp_responses_ = inp_responses;
        }
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_uniform_voting(uniform_voting_t voting) {
        this->uniform_voting_ = std::move(voting);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_classification_t<T>>
    auto& set_distance_voting(distance_voting_t voting) {
        this->distance_voting_ = std::move(voting);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    auto& set_uniform_regression(uniform_regression_t regression) {
        this->uniform_regression_ = std::move(regression);
        return *this;
    }

    template <typename T = Task, typename = detail::enable_if_regression_t<T>>
    auto& set_distance_regression(distance_regression_t regression) {
        this->distance_regression_ = std::move(regression);
        return *this;
    }

    auto& set_responses(const array<res_t>& responses) {
        if (result_options_.test(result_options::responses)) {
            ONEDAL_ASSERT(responses.get_count() == query_length_);
            this->responses_ = pr::ndarray<res_t, 1>::wrap_mutable(responses, query_length_);
        }
        return *this;
    }

    auto& set_indices(const array<idx_t>& indices) {
        if (result_options_.test(result_options::indices)) {
            ONEDAL_ASSERT(indices.get_count() ==
                          de::check_mul_overflow(query_length_, k_neighbors_));
            this->indices_ =
                pr::ndarray<idx_t, 2>::wrap_mutable(indices, { query_length_, k_neighbors_ });
        }
        return *this;
    }

    auto& set_distances(array<Float>& distances) {
        if (result_options_.test(result_options::distances)) {
            ONEDAL_ASSERT(distances.get_count() ==
                          de::check_mul_overflow(query_length_, k_neighbors_));
            this->distances_ =
                pr::ndarray<Float, 2>::wrap_mutable(distances, { query_length_, k_neighbors_ });
        }
        return *this;
    }

    auto get_blocking() const {
        return bk::uniform_blocking(query_length_, query_block_);
    }

    // Note: `inp_distances` can be modified if
    // metric is Euclidean
    sycl::event operator()(std::int64_t qb_id,
                           pr::ndview<idx_t, 2>& inp_indices,
                           pr::ndview<Float, 2>& inp_distances,
                           const bk::event_vector& deps = {}) {
        sycl::event copy_indices, copy_distances, comp_responses;
        const auto blocking = this->get_blocking();

        const auto from = blocking.get_block_start_index(qb_id);
        const auto to = blocking.get_block_end_index(qb_id);

        if (result_options_.test(result_options::indices)) {
            auto out_block = indices_.get_row_slice(from, to);
            copy_indices = copy(queue_, out_block, inp_indices, deps);
        }

        if (result_options_.test(result_options::distances)) {
            auto out_block = distances_.get_row_slice(from, to);
            if (this->compute_sqrt_) {
                copy_distances = copy_with_sqrt(queue_, inp_distances, out_block, deps);
            }
            else {
                copy_distances = copy(queue_, out_block, inp_distances, deps);
            }
        }

        if (result_options_.test(result_options::responses)) {
            using namespace bk;
            auto out_block = responses_.get_slice(from, to);
            const auto ndeps = deps + copy_indices + copy_distances;
            auto temp_resp = temp_resp_.get_row_slice(0, to - from);
            auto s_event = select_indexed(queue_, inp_indices, inp_responses_, temp_resp, ndeps);

            // One and only one functor can be initialized
            ONEDAL_ASSERT((bool(distance_voting_) + bool(uniform_voting_) +
                           bool(distance_regression_) + bool(uniform_regression_)) == 1);

            if constexpr (std::is_same_v<Task, task::classification>) {
                if (uniform_voting_) {
                    comp_responses = uniform_voting_->operator()(temp_resp, out_block, { s_event });
                }

                if (distance_voting_) {
                    sycl::event sqrt_event;

                    if (this->compute_sqrt_) {
                        sqrt_event = copy_with_sqrt(queue_, inp_distances, inp_distances, deps);
                    }

                    comp_responses = distance_voting_->operator()(temp_resp,
                                                                  inp_distances,
                                                                  out_block,
                                                                  { sqrt_event, s_event });
                }
            }

            if constexpr (std::is_same_v<Task, task::regression>) {
                if (uniform_regression_) {
                    comp_responses =
                        uniform_regression_->operator()(temp_resp, out_block, { s_event });
                }

                if (distance_regression_) {
                    sycl::event sqrt_event;

                    if (this->compute_sqrt_) {
                        sqrt_event = copy_with_sqrt(queue_, inp_distances, inp_distances, deps);
                    }

                    comp_responses = distance_regression_->operator()(temp_resp,
                                                                      inp_distances,
                                                                      out_block,
                                                                      { sqrt_event, s_event });
                }
            }
        }

        sycl::event::wait_and_throw({ copy_indices, copy_distances, comp_responses });
        return sycl::event();
    }

private:
    sycl::queue& queue_;
    const result_option_id result_options_;
    const std::int64_t query_block_, query_length_, k_neighbors_;
    pr::ndview<res_t, 1> inp_responses_;
    pr::ndarray<res_t, 2> temp_resp_;
    pr::ndarray<res_t, 1> responses_;
    pr::ndarray<Float, 2> distances_;
    pr::ndarray<idx_t, 2> indices_;
    uniform_voting_t uniform_voting_;
    distance_voting_t distance_voting_;
    uniform_regression_t uniform_regression_;
    distance_regression_t distance_regression_;
    bool compute_sqrt_ = false;
};

template <typename Float, typename Task>
static infer_result<Task> call_kernel(const context_gpu& ctx,
                                      const descriptor_t<Task>& desc,
                                      const table& infer,
                                      const model<Task>& m) {
    using res_t = response_t<Task>;

    auto distance_impl = detail::get_distance_impl(desc);
    if (!distance_impl) {
        throw internal_error{ de::error_messages::unknown_distance_type() };
    }
    else if (distance_impl->get_daal_distance_type() != daal_distance_t::minkowski) {
        throw internal_error{ de::error_messages::distance_is_not_supported_for_gpu() };
    }

    const bool is_euclidean_distance =
        (distance_impl->get_daal_distance_type() == daal_distance_t::minkowski) &&
        (distance_impl->get_degree() == 2.0);

    auto& queue = ctx.get_queue();
    bk::interop::execution_context_guard guard(queue);

    const auto trained_model = dynamic_cast_to_knn_model<Task, brute_force_model_impl<Task>>(m);
    const auto train = trained_model->get_data();
    const auto resps = trained_model->get_responses();

    const std::int64_t infer_row_count = infer.get_row_count();
    const std::int64_t feature_count = train.get_column_count();

    const std::int64_t class_count = desc.get_class_count();
    const std::int64_t neighbor_count = desc.get_neighbor_count();

    ONEDAL_ASSERT(train.get_column_count() == infer.get_column_count());

    auto arr_responses = array<res_t>{};
    if (desc.get_result_options().test(result_options::responses)) {
        arr_responses = array<res_t>::empty(queue, infer_row_count, sycl::usm::alloc::device);
    }
    auto arr_distances = array<Float>{};
    if (desc.get_result_options().test(result_options::distances) ||
        (desc.get_voting_mode() == voting_t::distance)) {
        const auto length = de::check_mul_overflow(infer_row_count, neighbor_count);
        arr_distances = array<Float>::empty(queue, length, sycl::usm::alloc::device);
    }
    auto arr_indices = array<idx_t>{};
    if (desc.get_result_options().test(result_options::indices)) {
        const auto length = de::check_mul_overflow(infer_row_count, neighbor_count);
        arr_indices = array<idx_t>::empty(queue, length, sycl::usm::alloc::device);
    }

    auto train_data = pr::table2ndarray<Float>(queue, train, sycl::usm::alloc::device);
    auto query_data = pr::table2ndarray<Float>(queue, infer, sycl::usm::alloc::device);
    auto resps_data = desc.get_result_options().test(result_options::responses)
                          ? pr::table2ndarray_1d<res_t>(queue, resps, sycl::usm::alloc::device)
                          : pr::ndarray<res_t, 1>{};

    const std::int64_t infer_block = pr::propose_query_block<Float>(queue, feature_count);
    const std::int64_t train_block = pr::propose_train_block<Float>(queue, feature_count);

    knn_callback<Float, Task> callback(queue,
                                       desc.get_result_options(),
                                       infer_block,
                                       infer_row_count,
                                       neighbor_count);

    callback.set_inp_responses(resps_data);
    callback.set_responses(arr_responses);
    callback.set_distances(arr_distances);
    callback.set_indices(arr_indices);

    if constexpr (std::is_same_v<Task, task::classification>) {
        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::uniform)) {
            callback.set_uniform_voting(
                std::move(pr::make_uniform_voting(queue, infer_block, neighbor_count)));
        }

        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::distance)) {
            callback.set_distance_voting(
                std::move(pr::make_distance_voting<Float>(queue, infer_block, class_count)));
        }
    }

    if constexpr (std::is_same_v<Task, task::regression>) {
        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::uniform)) {
            callback.set_uniform_regression(
                std::move(pr::make_uniform_regression<res_t>(queue, infer_block, neighbor_count)));
        }

        if (desc.get_result_options().test(result_options::responses) &&
            (desc.get_voting_mode() == voting_mode::distance)) {
            callback.set_distance_regression(
                std::move(pr::make_distance_regression<Float>(queue, infer_block, neighbor_count)));
        }
    }

    if (is_euclidean_distance) {
        using dst_t = pr::squared_l2_distance<Float>;
        using search_t = pr::search_engine<Float, dst_t>;

        callback.set_euclidean_distance(true);

        const dst_t dist{ queue };
        const search_t search{ queue, train_data, train_block, dist };
        search(query_data, callback, infer_block, neighbor_count).wait_and_throw();
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
        if constexpr (detail::is_not_search_v<Task>) {
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
    return call_kernel<Float, Task>(ctx, desc, input.get_data(), input.get_model());
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
template struct infer_kernel_gpu<float, method::brute_force, task::regression>;
template struct infer_kernel_gpu<double, method::brute_force, task::regression>;
template struct infer_kernel_gpu<float, method::brute_force, task::search>;
template struct infer_kernel_gpu<double, method::brute_force, task::search>;

} // namespace oneapi::dal::knn::backend
