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

#include "oneapi/dal/algo/newton_cg/common.hpp"
#include "oneapi/dal/exceptions.hpp"

namespace oneapi::dal::newton_cg::detail {

namespace v1 {

template <typename Task>
class descriptor_impl : public base {
public:
    explicit descriptor_impl(double tol = 1e-4, std::int64_t maxiter = 100)
            : tol(tol),
              maxiter(maxiter) {}
    double tol;
    std::int64_t maxiter;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
double descriptor_base<Task>::get_tolerance() const {
    return impl_->tol;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_max_iteration() const {
    return impl_->maxiter;
}

template <typename Task>
void descriptor_base<Task>::set_tolerance_impl(double tol) {
    using msg = dal::detail::error_messages;
    if (tol < 0) {
        throw domain_error(msg::conv_tol_lt_zero());
    }
    impl_->tol = tol;
}

template <typename Task>
void descriptor_base<Task>::set_max_iteration_impl(std::int64_t maxiter) {
    using msg = dal::detail::error_messages;
    if (maxiter < 0) {
        throw domain_error(msg::max_iteration_count_lt_zero());
    }
    impl_->maxiter = maxiter;
}

template class ONEDAL_EXPORT descriptor_base<task::compute>;

} // namespace v1

} // namespace oneapi::dal::newton_cg::detail
