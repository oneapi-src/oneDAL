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



namespace oneapi::dal::kmeans_init::backend {

namespace bk = dal::backend;
namespace pr = dal::backend::primitives;

/*template< typename Communicator, typename Float, typename Index>
sycl::event extract_and_share_by_index(sycl::queue& queue,
                                       Communicator& comm,
                                       std::int64_t index,
                                       ndview<Float, 1>& place,
                                       const ndview<Index, 1>& )*/

template <typename SearchObject, typename Float>
sycl::event find_local_closest(sycl::queue& queue,
                               const SearchObject& search_object,
                               const pr::ndview<Float, 2>& samples,
                               pr::ndview<Float, 1>& closest_distances,
                               const event_vector& deps = {}) {
    ONEDAL_ASSERT(samples.has_data());
    ONEDAL_ASSERT(closest_distances.has_mutable_data());
    const auto feature_count = samples.get_dimension(1);
    const auto sample_count = samples.get_dimension(0);

    ONEDAL_ASSERT(sample_count == closest_distances.get_count());
    auto query_block = pr::propose_query_block(queue, sample_count);

    const pr::uniform_blocking blocking(sample_count, query_block);
    auto callback = [&](auto qb_id, const auto& indices, const auto& distances, 
                            const event_vector& dependencies) -> sycl::event {
        ONEDAL_ASSERT(indices.has_data() && distances.has_data());
        ONEDAL_ASSERT(distances.get_dimension(1) == std::int64_t{ 1 });

        const auto last = blocking.get_block_end_index(qb_id);
        const auto first = blocking.get_block_start_index(qb_id);
        const auto block_length = blocking.get_block_length(qb_id);
        ONEDAL_ASSERT(distances.get_dimension(0) == block_length);

        const auto output_slice = closest_distances.get_slice(first, last);
        auto output_2d = output_slice.template reshape<1>({ block_length, 1 });

        return pr::copy(queue, output_2d, distances, dependencies);
    };

    return search_object(samples, callback, query_block, 1, deps);
}


template <typename Float, typename Method, typename Task>
compute_result<Task> compute_kernel_distr<Float, Method, Task>::operator()(
        const dal::backend::context_gpu& ctx,
        const detail::descriptor_base<Task>& params,
        const compute_input<Task>& input) const {
    auto& queue = ctx.get_queue();
    const auto& data_table = input.get_data();

    const auto seed = params.get_seed();
    const auto sample_count = data_table.get_row_count();
    const auto cluster_count = params.get_cluster_count();
    const auto feature_count = data_table.get_column_count();

    ONEDAL_ASSERT(0 < cluster_count);
    ONEDAL_ASSERT(cluster_count <= sample_count);

    constexpr auto alloc = sycl::usm::alloc::device;
    auto data = pr::table2ndarray<Float>(queue, data_table, alloc);
    auto centroids = pr::empty(queue, {cluster_count, feature_count}, alloc);




    return compute_result<Task>{};
}

template struct compute_kernel_distr<float, method::plus_plus_dense, task::init>;
template struct compute_kernel_distr<double, method::plus_plus_dense, task::init>;

} // namespace oneapi::dal::kmeans_init::backend
