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

#include "oneapi/dal/backend/primitives/distance/distance.hpp"
#include "oneapi/dal/backend/primitives/distance/squared_l2_distance_auxilary.hpp"

#include "oneapi/dal/backend/primitives/blas.hpp"
#include "oneapi/dal/backend/primitives/reduction.hpp"

namespace oneapi::dal::backend::primitives {

template<typename Float>
bool is_subset_of_rows(const ndview<Float, 2>& container, 
                       const ndview<Float, 2>& content) {
    bool result = true;
    result &= (container.get_dimension(1) == content.get_dimension(1));
    result &= (container.get_leading_stride() == content.get_leading_stride());
    // Check for the first rows of arrays
    const Float* container_fr = container.get_data();
    const Float* content_fr = content.get_data(); 
    // We can safely compare pointers
    // Check if content starts after container
    result &= (container_fr <= content_fr);
    // Now we can safely use difference
    const auto fr_diff = content_fr - container_fr;
    result &= (fr_diff % container.get_leading_stride() == 0);
    // Check for the last rows of arrays
    const Float* container_lr = container_fr + 
                    container.get_leading_stride() * container.get_dimension(0);
    const Float* content_lr = content_fr + 
                    content.get_leading_stride() * content.get_dimension(0);
    result &= (content_lr <= container_lr);
    return result;
}

template<typename Float>
class l2_helper {
    using helper_t = l2_helper<Float>;
    using helper_ptr_t = detail::unique<helper_t>;
    using norms_res_t = std::tuple<const array<Float>, sycl::event>;
public:
    static auto initialize(sycl::queue& q, 
                           const ndview<Float, 2>& inp,
                           const event_vector& deps = {}) {
        auto [res_array, res_event] = compute_squared_l2_norms(q, inp, deps);
        using res_t = std::tuple<helper_t*, sycl::event>;
        auto* res_helper_ptr = new helper_t(q, inp, res_array);
        return res_t(res_helper_ptr, res_event);
    }
    bool have_norms(const ndview<Float, 2>& inp) const {
        return is_subset_of_rows(input_, inp);
    }
    norms_res_t get_norms(const ndview<Float, 2>& inp,
                          const event_vector& deps) const {
        ONEDAL_ASSERT(have_norms(inp));
        const auto diff_rows = (inp.get_data() - input_.get_data()) % inp.get_leading_stride();
        const auto* from_ptr = norms_.get_data() + diff_rows;
        const auto res_array = array<Float>::wrap(from_ptr, 
                                        inp.get_dimension(0));
        auto res_event = q_.submit([&](sycl::handler& h) {
            h.depends_on(deps);
        });
        return { res_array, res_event };
    }
private:
    l2_helper(sycl::queue& q, 
              const ndview<Float, 2>& input, 
              const array<Float>& norms) :
        q_{ q }, 
        input_{ input }, 
        norms_{ norms } {};
    sycl::queue& q_;
    const ndview<Float, 2> input_;
    const array<Float> norms_;
};

template<typename Float>
bool is_initialized(const l2_helper<Float>* helper) {
    return (helper != nullptr);
}

template<typename Float>
sycl::event distance<Float, squared_l2_metric<Float>>::initialize_helper(helper_ptr_t& out_helper_ptr,
                                                                         const ndview<Float, 2>& inp1,
                                                                         const event_vector& deps) {
    ONEDAL_ASSERT(!is_initialized(helper.get()));
    auto [res_helper_ptr, res_event] = helper_t::initialize(q_, inp1, deps);
    out_helper_ptr = helper_ptr_t(res_helper_ptr);
    return res_event;
}

template<typename Float>
sycl::event distance<Float, squared_l2_metric<Float>>::initialize(const ndview<Float, 2>& inp1,
                                                                  const ndview<Float, 2>& inp2,
                                                                  const event_vector& deps) {
    auto init1_event = initialize_helper(helper1_, inp1, deps);
    auto init2_event = initialize_helper(helper2_, inp2, deps);
    return q_.submit([&](sycl::handler& h) {
        h.depends_on({init1_event, init2_event});
    });
}

template<typename Float>
auto distance<Float, squared_l2_metric<Float>>::get_norms(const helper_ptr_t& helper,
                                                          const ndview<Float, 2>& inp,
                                                          const event_vector& deps) const -> norms_res_t {
    if(is_initialized(helper.get()) && helper->have_norms(inp)) {
        auto [res_array, res_event] = helper->get_norms(inp, deps);
        return {res_array, res_event};
    }
    auto [res_array, res_event] = compute_squared_l2_norms(q_, inp, deps);
    return {res_array, res_event};
}

template<typename Float>
sycl::event distance<Float, squared_l2_metric<Float>>::operator() (const ndview<Float, 2>& inp1,
                                                                   const ndview<Float, 2>& inp2,
                                                                   ndview<Float, 2>& out,
                                                                   const event_vector& deps) const {
    auto [norms1_array, norms1_event] = get_norms(helper1_, inp1, deps);
    auto [norms2_array, norms2_event] = get_norms(helper2_, inp2, deps);
    const auto norms1_view = ndview<Float, 1>::wrap(norms1_array.get_data(), { norms1_array.get_count() });
    const auto norms2_view = ndview<Float, 1>::wrap(norms2_array.get_data(), { norms2_array.get_count() });
    auto scatter_event = scatter_2d(q_, norms1_view, norms2_view, out, {norms1_event, norms2_event});
    return compute_inner_product(q_, inp1, inp2, out, {scatter_event});
}

template<typename Float>
distance<Float, squared_l2_metric<Float>>::distance(sycl::queue& q) : 
                                            q_{ q }, helper1_{ nullptr }, helper2_{ nullptr } {}

template<typename Float>
distance<Float, squared_l2_metric<Float>>::~distance()  = default;

#define INSTANTIATE(F)                                                      \
    template class distance<F, squared_l2_metric<F>>;

INSTANTIATE(float);
INSTANTIATE(double);

#undef INSTANTIATE

} // namespace oneapi::dal::backend::primitives
