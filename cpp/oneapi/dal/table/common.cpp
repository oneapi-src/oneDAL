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

#include "oneapi/dal/table/common.hpp"
#include "oneapi/dal/exceptions.hpp"
#include "oneapi/dal/table/backend/empty_table_impl.hpp"

using std::int64_t;

namespace oneapi::dal {

class detail::table_metadata_impl {
public:
    virtual ~table_metadata_impl() {}

    virtual int64_t get_feature_count() const = 0;
    virtual const feature_type& get_feature_type(int64_t index) const = 0;
    virtual const data_type& get_data_type(int64_t index) const = 0;
};

namespace backend {
class empty_metadata_impl : public detail::table_metadata_impl {
public:
    int64_t get_feature_count() const override {
        return 0;
    }
    const feature_type& get_feature_type(int64_t) const override {
        throw domain_error("cannot get feature type from empty metadata");
    }
    const data_type& get_data_type(int64_t) const override {
        throw domain_error("cannot get data type from empty metadata");
    }
};

class simple_metadata_impl : public detail::table_metadata_impl {
public:
    simple_metadata_impl(const array<data_type>& dtypes, const array<feature_type>& ftypes)
            : dtypes_(dtypes),
              ftypes_(ftypes) {
        if (dtypes_.get_count() != ftypes_.get_count()) {
            throw out_of_range("data type and feature type arrays shall be the same size");
        }
    }

    int64_t get_feature_count() const override {
        return dtypes_.get_count();
    }
    const feature_type& get_feature_type(int64_t i) const override {
        if (!is_in_range(i)) {
            throw out_of_range("feature index is out of range");
        }
        return ftypes_[i];
    }
    const data_type& get_data_type(int64_t i) const override {
        if (!is_in_range(i)) {
            throw out_of_range("feature index is out of range");
        }
        return dtypes_[i];
    }

private:
    bool is_in_range(int64_t i) const {
        return i >= 0 && i < dtypes_.get_count();
    }

private:
    array<data_type> dtypes_;
    array<feature_type> ftypes_;
};
} // namespace backend

table_metadata::table_metadata() : impl_(new backend::empty_metadata_impl()) {}

table_metadata::table_metadata(const array<data_type>& dtypes, const array<feature_type>& ftypes)
        : impl_(new backend::simple_metadata_impl(dtypes, ftypes)) {}

int64_t table_metadata::get_feature_count() const {
    return impl_->get_feature_count();
}

const feature_type& table_metadata::get_feature_type(int64_t feature_index) const {
    return impl_->get_feature_type(feature_index);
}

const data_type& table_metadata::get_data_type(int64_t feature_index) const {
    return impl_->get_data_type(feature_index);
}

table::table() : table(backend::empty_table_impl{}) {}

table::table(table&& t) : impl_(std::move(t.impl_)) {
    using wrapper = detail::table_impl_wrapper<backend::empty_table_impl>;
    using wrapper_ptr = detail::shared<wrapper>;

    t.impl_ = wrapper_ptr(new wrapper(backend::empty_table_impl{}));
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

void table::init_impl(detail::table_impl_iface* impl) {
    impl_ = pimpl{ impl };
}

} // namespace oneapi::dal
