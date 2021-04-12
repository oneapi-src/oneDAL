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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/exceptions.hpp"
// #include "oneapi/dal/table/backend/empty_table_impl.hpp"

using std::int64_t;

namespace oneapi::dal {
namespace detail {
namespace v1 {

class table_metadata_impl {
public:
    virtual ~table_metadata_impl() = default;
    virtual int64_t get_feature_count() const = 0;
    virtual const feature_type& get_feature_type(int64_t index) const = 0;
    virtual const data_type& get_data_type(int64_t index) const = 0;
};

} // namespace v1
} // namespace detail

namespace v1 {

using detail::v1::table_metadata_impl;

class empty_metadata_impl : public table_metadata_impl, public base {
public:
    int64_t get_feature_count() const override {
        return 0;
    }

    const feature_type& get_feature_type(int64_t) const override {
        throw domain_error(
            dal::detail::error_messages::cannot_get_feature_type_from_empty_metadata());
    }

    const data_type& get_data_type(int64_t) const override {
        throw domain_error(dal::detail::error_messages::cannot_get_data_type_from_empty_metadata());
    }
};

class simple_metadata_impl : public table_metadata_impl, public base {
public:
    simple_metadata_impl(const array<data_type>& dtypes, const array<feature_type>& ftypes)
            : dtypes_(dtypes),
              ftypes_(ftypes) {
        if (dtypes_.get_count() != ftypes_.get_count()) {
            throw out_of_range{
                dal::detail::error_messages::
                    element_count_in_data_type_and_feature_type_arrays_does_not_match()
            };
        }
    }

    int64_t get_feature_count() const override {
        return dtypes_.get_count();
    }

    const feature_type& get_feature_type(int64_t i) const override {
        if (!is_in_range(i)) {
            throw out_of_range(dal::detail::error_messages::feature_index_is_out_of_range());
        }
        return ftypes_[i];
    }

    const data_type& get_data_type(int64_t i) const override {
        if (!is_in_range(i)) {
            throw out_of_range(dal::detail::error_messages::feature_index_is_out_of_range());
        }
        return dtypes_[i];
    }

private:
    bool is_in_range(int64_t i) const {
        return i >= 0 && i < dtypes_.get_count();
    }

    array<data_type> dtypes_;
    array<feature_type> ftypes_;
};

class empty_table_impl : public detail::table_iface, public base {
public:
    static constexpr std::int64_t pure_empty_table_kind = 0;

    std::int64_t get_column_count() const override {
        return 0;
    }

    std::int64_t get_row_count() const override {
        return 0;
    }

    std::int64_t get_kind() const override {
        return pure_empty_table_kind;
    }

    data_layout get_data_layout() const override {
        return data_layout::unknown;
    }

    const table_metadata& get_metadata() const override {
        static table_metadata metadata;
        return metadata;
    }

    detail::pull_rows_iface* get_pull_rows_iface() override {
        return nullptr;
    }

    detail::pull_column_iface* get_pull_column_iface() override {
        return nullptr;
    }
};

table_metadata::table_metadata() : impl_(new empty_metadata_impl()) {}

table_metadata::table_metadata(const array<data_type>& dtypes, const array<feature_type>& ftypes)
        : impl_(new simple_metadata_impl(dtypes, ftypes)) {}

int64_t table_metadata::get_feature_count() const {
    return impl_->get_feature_count();
}

const feature_type& table_metadata::get_feature_type(int64_t feature_index) const {
    return impl_->get_feature_type(feature_index);
}

const data_type& table_metadata::get_data_type(int64_t feature_index) const {
    return impl_->get_data_type(feature_index);
}

table::table() : table(new empty_table_impl{}) {}

table::table(table&& t) : impl_(std::move(t.impl_)) {
    t.impl_.reset(new empty_table_impl{});
}

table& table::operator=(table&& t) {
    this->impl_.swap(t.impl_);
    return *this;
}

bool table::has_data() const noexcept {
    return impl_->get_column_count() > 0 && impl_->get_row_count() > 0;
}

int64_t table::get_column_count() const {
    return impl_->get_column_count();
}

int64_t table::get_row_count() const {
    return impl_->get_row_count();
}

const table_metadata& table::get_metadata() const {
    return impl_->get_metadata();
}

int64_t table::get_kind() const {
    return impl_->get_kind();
}

data_layout table::get_data_layout() const {
    return impl_->get_data_layout();
}

} // namespace v1
} // namespace oneapi::dal
