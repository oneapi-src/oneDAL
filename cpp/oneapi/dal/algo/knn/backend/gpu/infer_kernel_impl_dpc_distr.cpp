// /*******************************************************************************
// * Copyright 2020 Intel Corporation
// *
// * Licensed under the Apache License, Version 2.0 (the "License");
// * you may not use this file except in compliance with the License.
// * You may obtain a copy of the License at
// *
// *     http://www.apache.org/licenses/LICENSE-2.0
// *
// * Unless required by applicable law or agreed to in writing, software
// * distributed under the License is distributed on an "AS IS" BASIS,
// * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// * See the License for the specific language governing permissions and
// * limitations under the License.
// *******************************************************************************/

// #include "oneapi/dal/backend/interop/common_dpc.hpp"
// #include "oneapi/dal/backend/interop/error_converter.hpp"
// #include "oneapi/dal/backend/interop/table_conversion.hpp"

// #include "oneapi/dal/algo/knn/backend/model_conversion.hpp"
// #include "oneapi/dal/algo/knn/backend/gpu/infer_kernel.hpp"
// #include "oneapi/dal/algo/knn/backend/gpu/infer_kernel_impl.hpp"

// #include "oneapi/dal/algo/knn/backend/distance_impl.hpp"
// #include "oneapi/dal/algo/knn/backend/model_impl.hpp"
// #include "oneapi/dal/detail/policy.hpp"
// #include "oneapi/dal/backend/primitives/common.hpp"
// #include "oneapi/dal/backend/primitives/ndarray.hpp"
// #include "oneapi/dal/backend/primitives/regression.hpp"
// #include "oneapi/dal/backend/primitives/search.hpp"
// #include "oneapi/dal/backend/primitives/selection.hpp"
// #include "oneapi/dal/backend/primitives/voting.hpp"
// #include "oneapi/dal/backend/primitives/utils.hpp"
// #include "oneapi/dal/backend/communicator.hpp"

// #include "oneapi/dal/table/row_accessor.hpp"

// #include "oneapi/dal/detail/common.hpp"

// namespace oneapi::dal::knn::backend {

// // template <typename T1, typename T2>
// // inline sycl::event copy_with_sqrt(sycl::queue& q,
// //                            const pr::ndview<T2, 2>& src,
// //                            pr::ndview<T1, 2>& dst,
// //                            const bk::event_vector& deps = {}) {
// //     static_assert(de::is_floating_point<T1>());
// //     static_assert(de::is_floating_point<T2>());
// //     ONEDAL_ASSERT(src.has_data());
// //     ONEDAL_ASSERT(dst.has_mutable_data());
// //     const pr::ndshape<2> dst_shape = dst.get_shape();
// //     ONEDAL_ASSERT(dst_shape == src.get_shape());
// //     T1* const dst_ptr = dst.get_mutable_data();
// //     const T2* const src_ptr = src.get_data();
// //     const auto dst_stride = dst.get_leading_stride();
// //     const auto src_stride = src.get_leading_stride();
// //     const auto cp_range = bk::make_range_2d(dst_shape[0], dst_shape[1]);
// //     return q.submit([&](sycl::handler& h) {
// //         h.depends_on(deps);
// //         h.parallel_for(cp_range, [=](sycl::id<2> idx) {
// //             T1& dst_ref = *(dst_ptr + idx[0] * dst_stride + idx[1]);
// //             const T2& val_ref = *(src_ptr + idx[0] * src_stride + idx[1]);
// //             dst_ref = sycl::sqrt(val_ref);
// //         });
// //     });
// // }

// template <typename Float, typename Task>
// class knn_callback_distr {
//     using dst_t = Float;
//     using idx_t = std::int64_t;
//     using res_t = response_t<Task, Float>;
//     using comm_t = bk::communicator<spmd::device_memory_access::usm>;

//     using uniform_voting_t = std::unique_ptr<pr::uniform_voting<res_t>>;
//     using distance_voting_t = std::unique_ptr<pr::distance_voting<dst_t>>;
//     using uniform_regression_t = std::unique_ptr<pr::uniform_regression<res_t>>;
//     using distance_regression_t = std::unique_ptr<pr::distance_regression<dst_t>>;

// public:
//     knn_callback_distr(sycl::queue& q,
//                  comm_t c,
//                  result_option_id results,
//                  std::int64_t query_block,
//                  std::int64_t query_length,
//                  std::int64_t k_neighbors)
//             : queue_(q),
//               comm_(c),
//               result_options_(results),
//               query_block_(query_block),
//               query_length_(query_length),
//               k_neighbors_(k_neighbors) {
//         if (result_options_.test(result_options::responses)) {
//             this->temp_resp_ = pr::ndarray<res_t, 2>::empty(q,
//                                                             { query_block, k_neighbors },
//                                                             sycl::usm::alloc::device);
//         }
//     }

//     auto& set_euclidean_distance(bool is_euclidean_distance) {
//         this->compute_sqrt_ = is_euclidean_distance;
//         return *this;
//     }

//     auto& set_inp_responses(const pr::ndview<res_t, 1>& inp_responses) {
//         if (result_options_.test(result_options::responses)) {
//             this->inp_responses_ = inp_responses;
//         }
//         return *this;
//     }

//     template <typename T = Task, typename = detail::enable_if_classification_t<T>>
//     auto& set_uniform_voting(uniform_voting_t voting) {
//         this->uniform_voting_ = std::move(voting);
//         return *this;
//     }

//     template <typename T = Task, typename = detail::enable_if_classification_t<T>>
//     auto& set_distance_voting(distance_voting_t voting) {
//         this->distance_voting_ = std::move(voting);
//         return *this;
//     }

//     template <typename T = Task, typename = detail::enable_if_regression_t<T>>
//     auto& set_uniform_regression(uniform_regression_t regression) {
//         this->uniform_regression_ = std::move(regression);
//         return *this;
//     }

//     template <typename T = Task, typename = detail::enable_if_regression_t<T>>
//     auto& set_distance_regression(distance_regression_t regression) {
//         this->distance_regression_ = std::move(regression);
//         return *this;
//     }

//     auto& set_responses(const pr::ndview<res_t, 1>& responses) {
//         if (result_options_.test(result_options::responses)) {
//             ONEDAL_ASSERT(responses.get_count() == query_length_);
//             this->responses_ = responses;
//         }
//         return *this;
//     }

//     auto& set_indices(const pr::ndview<idx_t, 2>& indices) {
//         if (result_options_.test(result_options::indices)) {
//             ONEDAL_ASSERT(indices.get_dimension(0) == query_length_);
//             ONEDAL_ASSERT(indices.get_dimension(1) == k_neighbors_);
//             this->indices_ = indices;
//         }
//         return *this;
//     }

//     auto& set_part_indices(const pr::ndview<idx_t, 2>& part_indices) {
//         if (result_options_.test(result_options::indices)) {
//             ONEDAL_ASSERT(part_indices.get_dimension(0) == 2 * query_length_);
//             ONEDAL_ASSERT(part_indices.get_dimension(1) == k_neighbors_);
//             this->part_indices_ = part_indices;
//         }
//         return *this;
//     }

//     auto& set_distances(const pr::ndview<dst_t, 2>& distances) {
//         if (result_options_.test(result_options::distances)) {
//             ONEDAL_ASSERT(distances.get_dimension(0) == query_length_);
//             ONEDAL_ASSERT(distances.get_dimension(1) == k_neighbors_);
//             this->distances_ = distances;
//         }
//         return *this;
//     }

//     auto& set_part_distances(const pr::ndview<dst_t, 2>& part_distances) {
//         if (result_options_.test(result_options::distances)) {
//             ONEDAL_ASSERT(part_distances.get_dimension(0) == 2 * query_length_);
//             ONEDAL_ASSERT(part_distances.get_dimension(1) == k_neighbors_);
//             this->part_distances_ = part_distances;
//         }
//         return *this;
//     }

//     sycl::event reset_dists_inds(const bk::event_vector& deps) const {
//         constexpr auto default_dst_value = limits<Float>::max();
//         constexpr auto default_idx_value = -1;
//         auto out_dsts = fill(this->queue_, this->distances_, default_dst_value, deps);
//         auto out_idcs = fill(this->queue_, this->indices_, default_idx_value, out_dsts);
//         return out_idcs;
//     }

//     auto& set_global_index_offset(int64_t offset) {
//         global_index_offset_ = offset;
//         return *this;
//     }

//     auto& set_last_iteration(bool last_iteration) {
//         this->last_iteration_ = last_iteration;
//         return *this;
//     }

//     sycl::event finalize(std::int64_t qb_id,
//                          pr::ndview<idx_t, 2>& inp_indices,
//                          pr::ndview<Float, 2>& inp_distances,
//                          const bk::event_vector& deps = {}) {
//         //TODO: figure out how to handle this function (inputs, etc.)
//         const auto bounds = this->block_bounds(qb_id);

//         if (result_options_.test(result_options::indices)) {
//             copy_indices = this->output_indices(bounds, inp_indices, deps);
//         }

//         if (result_options_.test(result_options::distances)) {
//             copy_distances = this->output_distances(bounds, inp_distances, deps);
//         }
//         if (result_options_.test(result_options::responses)) {
//             using namespace bk;
//             const auto ndeps = deps + copy_indices + copy_distances;
//             comp_responses = this->output_responses(bounds, inp_indices, inp_distances, ndeps);
//         }

//         sycl::event::wait_and_throw({ copy_indices, copy_distances, comp_responses });
//         return sycl::event();
//     }

//     sycl::event operator()(std::int64_t qb_id,
//                            pr::ndview<idx_t, 2>& inp_indices,
//                            pr::ndview<Float, 2>& inp_distances,
//                            const bk::event_vector& deps = {}) {
//         sycl::event copy_actual_dist_event, copy_current_dist_event, copy_actual_indc_event, copy_current_indc_event;
//         const auto& [first, last] = this->block_bounds(qb_id);

//         auto start_index = 2 * first;
//         auto middle_index = start_index + (last - first);
//         auto end_index = 2 * last;

//         //TODO: add assertions/checks - mostly related to ensuring things are size k
//         kselect_by_rows<Float> select = create_selection_objects(2 * k_neighbors_, k_neighbors_);
//         //TODO: how to handle setting responses (first thought is have flag in this function if last iter) - use select_indexed()
//         auto min_dist_dest = distances_.get_row_slice(first, last);
//         auto min_indc_dest = indices_.get_row_slice(first, last);

//         // add global offset value to input indices
//         ONEDAL_ASSERT(global_index_offset_ != -1);
//         auto treat_event = treat_indices(inp_indices, global_index_offset_, deps);

//         auto actual_min_dist_copy_dest = part_distances_.get_row_slice(start_index, middle_index);
//         auto current_min_dist_dest = part_distances_.get_row_slice(middle_index, end_index);
//         copy_actual_dist_event = pr::copy(queue_, actual_min_dist_copy_dest, min_dist_dest, treat_event);
//         copy_current_dist_event = pr::copy(queue_, current_min_dist_dest, inp_distances, treat_event);
        
//         auto actual_min_indc_copy_dest = part_indices_.get_row_slice(start_index, middle_index);
//         auto current_min_indc_dest = part_indices_.get_row_slice(middle_index, end_index);
//         copy_actual_indc_event = pr::copy(queue_, actual_min_indc_copy_dest, min_indc_dest, treat_event);
//         copy_current_indc_event = pr::copy(queue_, current_min_indc_dest, inp_indices, treat_event);

//         auto kselect_block = part_distances_.get_row_slice(first, last);
//         auto selt_event = select(queue_,
//                                 kselect_block,
//                                 k_neighbors,
//                                 min_dist_dest,
//                                 min_indc_dest,
//                                 { copy_actual_dist_event, copy_current_dist_event, copy_actual_indc_event, copy_current_indc_event });
//         auto final_event = select_indexed(part_indices_,
//                                          min_indc_dest,
//                                          { selt_event });
//         if (last_iteration_) { //calls same methods as original operator, using already set indices_ and distances_
//             final_event = finalize(qb_id, indices_, distances_, { final_event })
//         }
//         return final_event;
//     }

// protected:
//     auto get_blocking() const {
//         return bk::uniform_blocking(query_length_, query_block_);
//     }

//     auto block_bounds(std::int64_t qb_id) const {
//         const auto blocking = this->get_blocking();
//         const auto first = blocking.get_block_start_index(qb_id);
//         const auto last = blocking.get_block_end_index(qb_id);
//         return std::make_pair(first, last);
//     }

//     sycl::event output_distances(const std::pair<idx_t, idx_t>& bnds,
//                                  const pr::ndview<dst_t, 2>& inp_dts,
//                                  const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(inp_dts.has_data());
//         ONEDAL_ASSERT(this->result_options_.test(result_options::distances));

//         const auto& [first, last] = bnds;
//         ONEDAL_ASSERT(last > first);
//         auto& queue = this->queue_;

//         auto out_dts = this->distances_.get_row_slice(first, last);
//         ONEDAL_ASSERT((last - first) == inp_dts.get_dimension(0));
//         ONEDAL_ASSERT((last - first) == out_dts.get_dimension(0));

//         // Generally !csqrt is more probable
//         const bool& csqrt = this->compute_sqrt_;
//         if (!csqrt)
//             return pr::copy(queue, out_dts, inp_dts, deps);
//         else
//             return copy_with_sqrt(queue, inp_dts, out_dts, deps);
//     }

//     sycl::event output_indices(const std::pair<idx_t, idx_t>& bnds,
//                                const pr::ndview<idx_t, 2>& inp_ids,
//                                const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(inp_ids.has_data());
//         ONEDAL_ASSERT(this->result_options_.test(result_options::indices));

//         const auto& [first, last] = bnds;
//         ONEDAL_ASSERT(last > first);
//         auto& queue = this->queue_;

//         auto out_ids = this->indices_.get_row_slice(first, last);
//         ONEDAL_ASSERT((last - first) == inp_ids.get_dimension(0));
//         ONEDAL_ASSERT((last - first) == out_ids.get_dimension(0));
//         ONEDAL_ASSERT(inp_ids.get_shape() == out_ids.get_shape());

//         return pr::copy(queue, out_ids, inp_ids, deps);
//     }

//     template <typename T = Task, typename = detail::enable_if_classification_t<T>>
//     sycl::event do_ucls(const std::pair<idx_t, idx_t>& bnds,
//                         const pr::ndview<res_t, 2>& tmp_rps,
//                         const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(tmp_rps.has_data());
//         ONEDAL_ASSERT(bool(this->uniform_voting_));
//         ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

//         const auto& [first, last] = bnds;
//         ONEDAL_ASSERT(last > first);

//         auto out_rps = this->responses_.get_slice(first, last);
//         ONEDAL_ASSERT((last - first) == out_rps.get_count());
//         return (*(this->uniform_voting_))(tmp_rps, out_rps, deps);
//     }

//     template <typename T = Task, typename = detail::enable_if_regression_t<T>>
//     sycl::event do_ureg(const std::pair<idx_t, idx_t>& bnds,
//                         const pr::ndview<res_t, 2>& tmp_rps,
//                         const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(tmp_rps.has_data());
//         ONEDAL_ASSERT(bool(this->uniform_regression_));
//         ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

//         const auto& [first, last] = bnds;
//         ONEDAL_ASSERT(last > first);

//         auto out_rps = this->responses_.get_slice(first, last);
//         ONEDAL_ASSERT((last - first) == out_rps.get_count());
//         return (*(this->uniform_regression_))(tmp_rps, out_rps, deps);
//     }

//     template <typename T = Task, typename = detail::enable_if_classification_t<T>>
//     sycl::event do_dcls(const std::pair<idx_t, idx_t>& bnds,
//                         const pr::ndview<res_t, 2>& tmp_rps,
//                         pr::ndview<dst_t, 2>& inp_dts,
//                         const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(inp_dts.has_data());
//         ONEDAL_ASSERT(tmp_rps.has_mutable_data());
//         ONEDAL_ASSERT(bool(this->distance_voting_));
//         ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

//         const auto& [first, last] = bnds;
//         ONEDAL_ASSERT(last > first);
//         auto& queue = this->queue_;

//         bk::event_vector ndeps{ deps.cbegin(), deps.cend() };
//         auto sq_event = copy_with_sqrt(queue, inp_dts, inp_dts, deps);
//         if (this->compute_sqrt_)
//             ndeps.push_back(sq_event);

//         auto out_rps = this->responses_.get_slice(first, last);
//         ONEDAL_ASSERT((last - first) == out_rps.get_count());
//         return (*(this->distance_voting_))(tmp_rps, inp_dts, out_rps, ndeps);
//     }

//     template <typename T = Task, typename = detail::enable_if_regression_t<T>>
//     sycl::event do_dreg(const std::pair<idx_t, idx_t>& bnds,
//                         const pr::ndview<res_t, 2>& tmp_rps,
//                         pr::ndview<dst_t, 2>& inp_dts,
//                         const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(inp_dts.has_data());
//         ONEDAL_ASSERT(tmp_rps.has_mutable_data());
//         ONEDAL_ASSERT(bool(this->distance_regression_));
//         ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

//         const auto& [first, last] = bnds;
//         ONEDAL_ASSERT(last > first);
//         auto& queue = this->queue_;

//         bk::event_vector ndeps{ deps.cbegin(), deps.cend() };
//         auto sq_event = copy_with_sqrt(queue, inp_dts, inp_dts, deps);
//         if (this->compute_sqrt_)
//             ndeps.push_back(sq_event);

//         auto out_rps = this->responses_.get_slice(first, last);
//         ONEDAL_ASSERT((last - first) == out_rps.get_count());
//         return (*(this->distance_regression_))(tmp_rps, inp_dts, out_rps, ndeps);
//     }

//     sycl::event output_responses(const std::pair<idx_t, idx_t>& bnds,
//                                  const pr::ndview<idx_t, 2>& inp_ids,
//                                  pr::ndview<dst_t, 2>& inp_dts,
//                                  const bk::event_vector& deps = {}) {
//         ONEDAL_ASSERT(inp_ids.has_data());
//         ONEDAL_ASSERT(this->result_options_.test(result_options::responses));

//         const auto& [first, last] = bnds;
//         const auto len = last - first;
//         ONEDAL_ASSERT(last > first);
//         auto& queue = this->queue_;

//         auto tmp_rps = this->temp_resp_.get_row_slice(0, len);

//         const auto& inp_rps = this->inp_responses_;
//         auto s_evt = pr::select_indexed(queue, inp_ids, inp_rps, tmp_rps, deps);

//         if constexpr (std::is_same_v<Task, task::classification>) {
//             const auto ucls = bool(this->uniform_voting_);
//             if (ucls)
//                 return this->do_ucls(bnds, tmp_rps, { s_evt });

//             const auto dcls = bool(this->distance_voting_);
//             if (dcls)
//                 return this->do_dcls(bnds, tmp_rps, inp_dts, { s_evt });
//         }

//         if constexpr (std::is_same_v<Task, task::regression>) {
//             const auto ureg = bool(this->uniform_regression_);
//             if (ureg)
//                 return this->do_ureg(bnds, tmp_rps, { s_evt });

//             const auto dreg = bool(this->distance_regression_);
//             if (dreg)
//                 return this->do_dreg(bnds, tmp_rps, inp_dts, { s_evt });
//         }

//         ONEDAL_ASSERT(false);
//         return sycl::event();
//     }

// private:
//     sycl::queue& queue_;
//     comm_t comm_;
//     const result_option_id result_options_;
//     const std::int64_t query_block_, query_length_, k_neighbors_;
//     pr::ndview<res_t, 1> inp_responses_;
//     pr::ndarray<res_t, 2> temp_resp_;
//     pr::ndview<res_t, 1> responses_;
//     pr::ndview<Float, 2> distances_;
//     pr::ndview<Float, 2> part_distances_;
//     pr::ndview<idx_t, 2> indices_;
//     pr::ndview<idx_t, 2> part_indices_;
//     int64_t global_index_offset_ = -1;
//     uniform_voting_t uniform_voting_;
//     distance_voting_t distance_voting_;
//     uniform_regression_t uniform_regression_;
//     distance_regression_t distance_regression_;
//     bool compute_sqrt_ = false;
//     bool last_iteration_ = false;
// };

// template <typename Task, typename Float, pr::ndorder torder, pr::ndorder qorder, typename RespT, bool cm_train>
// sycl::event bf_kernel_distr(sycl::queue& queue,
//                       bk::communicator<spmd::device_memory_access::usm> comm,
//                       const descriptor_t<Task>& desc,
//                       const table& train,
//                       const pr::ndview<Float, 2, qorder>& query,
//                       const pr::ndview<RespT, 1>& tresps,
//                       pr::ndview<Float, 2>& distances,
//                       pr::ndview<Float, 2>& part_distances,
//                       pr::ndview<idx_t, 2>& indices,
//                       pr::ndview<idx_t, 2>& part_indices,
//                       pr::ndview<RespT, 1>& qresps,
//                       const bk::event_vector& deps) {
//     using res_t = response_t<Task, Float>;

//     // Input arrays test section
//     ONEDAL_ASSERT(train.has_data());
//     ONEDAL_ASSERT(query.has_data());
//     [[maybe_unused]] const auto tcount = train.get_row_count();
//     const auto qcount = query.get_dimension(0);
//     const auto fcount = train.get_column_count();
//     ONEDAL_ASSERT(fcount == query.get_dimension(1));

//     // Output arrays test section
//     const auto& ropts = desc.get_result_options();
//     if (ropts.test(result_options::responses)) {
//         ONEDAL_ASSERT(tresps.has_data());
//         ONEDAL_ASSERT(qresps.has_mutable_data());
//         ONEDAL_ASSERT(tcount == tresps.get_count());
//         ONEDAL_ASSERT(qcount == qresps.get_count());
//     }
//     const auto kcount = desc.get_neighbor_count();
//     if (ropts.test(result_options::indices)) {
//         ONEDAL_ASSERT(indices.has_mutable_data());
//         ONEDAL_ASSERT(qcount == indices.get_dimension(0));
//         ONEDAL_ASSERT(kcount == indices.get_dimension(1));
//     }
//     if (ropts.test(result_options::distances)) {
//         ONEDAL_ASSERT(distances.has_mutable_data());
//         ONEDAL_ASSERT(qcount == distances.get_dimension(0));
//         ONEDAL_ASSERT(kcount == distances.get_dimension(1));
//     }

//     auto block_size = get_block_size();
//     auto rank_count = comm.get_rank_count();
//     auto node_sample_counts = pr::ndarray<std::int64_t, 1>::empty({ rank_count });
//     // ONEDAL_PROFILER_TASK needed?
//     comm.allgather(tcount, node_sample_counts.flatten()).wait();

//     auto current_rank = comm.get_rank();
//     auto prev_node = (current_rank - 1) % rank_count;
//     auto next_node = (current_rank + 1) % rank_count;

//     auto [boundaries, nodes] = get_boundary_indices(node_sample_counts, block_size);

//     auto block_count = nodes.size();

//     auto train_block_queue = split_dataset(queue, train, block_size, deps);

//     const auto qbcount = pr::propose_query_block<Float>(queue, fcount);
//     const auto tbcount = pr::propose_train_block<Float>(queue, fcount);

//     knn_callback_distr<Float, Task> callback(queue, comm, ropts, qbcount, qcount, kcount);
    
//     callback.set_inp_responses(tresps);
//     callback.set_distances(distances);
//     callback.set_part_distances(part_distances);
//     callback.set_responses(qresps);
//     callback.set_indices(indices);
//     callback.set_part_indices(part_indices);

//     auto next_event = callback.reset_dists_inds(deps);

//     if constexpr (std::is_same_v<Task, task::classification>) {
//         if (desc.get_result_options().test(result_options::responses) &&
//             (desc.get_voting_mode() == voting_mode::uniform)) {
//             callback.set_uniform_voting(std::move(pr::make_uniform_voting(queue, qbcount, kcount)));
//         }

//         if (desc.get_result_options().test(result_options::responses) &&
//             (desc.get_voting_mode() == voting_mode::distance)) {
//             callback.set_distance_voting(
//                 std::move(pr::make_distance_voting<Float>(queue, qbcount, kcount)));
//         }
//     }

//     if constexpr (std::is_same_v<Task, task::regression>) {
//         if (desc.get_result_options().test(result_options::responses) &&
//             (desc.get_voting_mode() == voting_mode::uniform)) {
//             callback.set_uniform_regression(
//                 std::move(pr::make_uniform_regression<res_t>(queue, qbcount, kcount)));
//         }

//         if (desc.get_result_options().test(result_options::responses) &&
//             (desc.get_voting_mode() == voting_mode::distance)) {
//             callback.set_distance_regression(
//                 std::move(pr::make_distance_regression<Float>(queue, qbcount, kcount)));
//         }
//     }

//     auto distance_impl = detail::get_distance_impl(desc);
//     if (!distance_impl) {
//         throw internal_error{ de::error_messages::unknown_distance_type() };
//     }

//     using daal_distance_t = decltype(distance_impl->get_daal_distance_type());
//     const bool is_minkowski_distance =
//         distance_impl->get_daal_distance_type() == daal_distance_t::minkowski;
//     const bool is_chebyshev_distance =
//         distance_impl->get_daal_distance_type() == daal_distance_t::chebyshev;
//     const bool is_cosine_distance =
//         distance_impl->get_daal_distance_type() == daal_distance_t::cosine;
//     const bool is_euclidean_distance =
//         is_minkowski_distance && (distance_impl->get_degree() == 2.0);

//     if (is_cosine_distance) {
//         using dst_t = pr::cosine_distance<Float>;
//         using search_t = pr::search_engine<Float, dst_t, torder>;

//         const dst_t dist{ queue };
//         const search_t search{ queue, dist };
//     }

//     if (is_chebyshev_distance) {
//         using dst_t = pr::chebyshev_distance<Float>;
//         using search_t = pr::search_engine<Float, dst_t, torder>;

//         const dst_t dist{ queue };
//         const search_t search{ queue, dist };
//     }

//     if (is_euclidean_distance) {
//         using dst_t = pr::squared_l2_distance<Float>;
//         using search_t = pr::search_engine<Float, dst_t, torder>;

//         callback.set_euclidean_distance(true);

//         const dst_t dist{ queue };
//         const search_t search{ queue, dist };
//     }
//     else if (is_minkowski_distance) {
//         using met_t = pr::lp_metric<Float>;
//         using dst_t = pr::lp_distance<Float>;
//         using search_t = pr::search_engine<Float, dst_t, torder>;

//         const dst_t dist{ queue, met_t(distance_impl->get_degree()) };
//         const search_t search{ queue, dist };
//     }

//     const auto first_block_index = std::find(nodes.begin(), nodes.end(), current_rank);
//     ONEDAL_ASSERT(first_block_index != nodes.end());
    
//     for(std::int32_t block_number = 0; block_number < block_count; block_number++) {
//         // TODO: revise variable names? specifically block_count, block_index, block_number, block_size
//         auto current_block = train_block_queue.pop_front();
//         auto block_index = (block_number + first_block_index) % block_count;
//         auto actual_rows_in_block = boundaries.at(block_index + 1) - boundaries.at(block_index);

//         auto sc = current_block.get_dimension(0);
//         ONEDAL_ASSERT(sc >= actual_rows_in_block);
//         auto curr_k = std::min(actual_rows_in_block, k);
//         auto actual_current_block = current_block.get_row_slice(0, actual_rows_in_block);

//         callback.set_global_index_offset(boundaries.at(block_index));
//         if (block_number == block_count - 1) {
//             callback.set_last_iteration(true);
//         }
        
//         search.reset_train_data(actual_current_block, tbcount);
//         next_event = search(query, callback, qbcount, kcount, { next_event });
    
//         comm.sendrecv_replace(current_block, prev_node, next_node).wait();
//     }

//     return next_event;
// }

// #define INSTANTIATE(T, I, R, F, A, B)                                                       \
//     template sycl::event bf_kernel_distr(sycl::queue&,                                      \
//                                    bk::communicator<spmd::device_memory_access::usm>,       \
//                                    const descriptor_t<T>&,                                  \
//                                    const pr::ndview<F, 2, A>&,                              \
//                                    const pr::ndview<F, 2, B>&,                              \
//                                    const pr::ndview<R, 1>&,                                 \
//                                    pr::ndview<F, 2>&,                                       \
//                                    pr::ndview<F, 2>&,                                       \
//                                    pr::ndview<I, 2>&,                                       \
//                                    pr::ndview<I, 2>&,                                       \
//                                    pr::ndview<R, 1>&,                                       \
//                                    const bk::event_vector&);

// #define INSTANTIATE_B(T, I, R, F, A)           \
//     INSTANTIATE(T, I, R, F, A, pr::ndorder::c) \
//     INSTANTIATE(T, I, R, F, A, pr::ndorder::f)

// #define INSTANTIATE_A(T, I, R, F)             \
//     INSTANTIATE_B(T, I, R, F, pr::ndorder::c) \
//     INSTANTIATE_B(T, I, R, F, pr::ndorder::f)

// #define INSTANTIATE_T(I, F)                                 \
//     INSTANTIATE_A(task::classification, I, std::int32_t, F) \
//     INSTANTIATE_A(task::regression, I, float, F)            \
//     INSTANTIATE_A(task::search, I, int, F)

// #define INSTANTIATE_F(I)    \
//     INSTANTIATE_T(I, float) \
//     INSTANTIATE_T(I, double)

// INSTANTIATE_F(std::int32_t)

// } // namespace oneapi::dal::knn::backend
