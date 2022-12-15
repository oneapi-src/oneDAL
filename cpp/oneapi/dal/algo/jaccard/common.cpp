/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "oneapi/dal/algo/jaccard/common.hpp"

namespace oneapi::dal::preview::jaccard {
namespace detail {

template <typename Task>
class descriptor_impl : public base {
public:
    std::int64_t row_range_begin = 0;
    std::int64_t row_range_end = 0;
    std::int64_t column_range_begin = 0;
    std::int64_t column_range_end = 0;
};

template <typename Task>
descriptor_base<Task>::descriptor_base() : impl_(new descriptor_impl<Task>{}) {}

template <typename Task>
std::int64_t descriptor_base<Task>::get_row_range_begin() const {
    return impl_->row_range_begin;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_row_range_end() const {
    return impl_->row_range_end;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_column_range_begin() const {
    return impl_->column_range_begin;
}

template <typename Task>
std::int64_t descriptor_base<Task>::get_column_range_end() const {
    return impl_->column_range_end;
}

template <typename Task>
void descriptor_base<Task>::set_row_range_impl(std::int64_t begin, std::int64_t end) {
    impl_->row_range_begin = begin;
    impl_->row_range_end = end;
}

template <typename Task>
void descriptor_base<Task>::set_column_range_impl(std::int64_t begin, std::int64_t end) {
    impl_->column_range_begin = begin;
    impl_->column_range_end = end;
}

template <typename Task>
void descriptor_base<Task>::set_block_impl(
    const std::initializer_list<std::int64_t>& row_range,
    const std::initializer_list<std::int64_t>& column_range) {
    impl_->row_range_begin = *row_range.begin();
    impl_->row_range_end = *(row_range.begin() + 1);
    impl_->column_range_begin = *column_range.begin();
    impl_->column_range_end = *(column_range.begin() + 1);
}

template class ONEDAL_EXPORT descriptor_base<task::all_vertex_pairs>;
} // namespace detail

void* caching_builder::operator()(std::int64_t block_max_size) {
    if (size < block_max_size) {
        size = block_max_size;
        result_ptr.reset();
        result_ptr =
            std::shared_ptr<byte_t>(new byte_t[block_max_size], std::default_delete<byte_t[]>());
    }
    return static_cast<void*>(result_ptr.get());
}

} // namespace oneapi::dal::preview::jaccard
