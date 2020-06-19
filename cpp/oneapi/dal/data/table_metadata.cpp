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

#include "onedal/table_metadata.hpp"

using std::int64_t;

namespace dal {
namespace detail {

struct table_feature_impl {
    data_type dtype;
    feature_type ftype;

    table_feature_impl(data_type dt, feature_type ft)
        : dtype(dt),
          ftype(ft) {}
};

class table_metadata_impl {
public:
    virtual ~table_metadata_impl() {}

    virtual int64_t get_feature_count() const = 0;
    virtual const table_feature& get_feature(int64_t feature_index) const = 0;
};

class empty_metadata_impl : public table_metadata_impl {
public:
    int64_t get_feature_count() const override {
        return 0;
    }

    const table_feature& get_feature(int64_t feature_index) const {
        throw std::runtime_error("no features in empty table");
    }
};

class simple_metadata_impl : public table_metadata_impl {
public:
    simple_metadata_impl(array<table_feature> features)
        : features_(features) {}

    int64_t get_feature_count() const override {
        return features_.get_size();
    }

    const table_feature& get_feature(int64_t feature_index) const override {
        return features_[feature_index];
    }

private:
    array<table_feature> features_;
};

class homogen_table_metadata_impl : public table_metadata_impl {
public:
    homogen_table_metadata_impl()
        : feature_count_(0) {}

    homogen_table_metadata_impl(const table_feature& feature,
                                homogen_data_layout layout,
                                std::int64_t feature_count)
        : feature_(feature),
          layout_(layout),
          feature_count_(feature_count) {}

    homogen_data_layout get_data_layout() const {
        return layout_;
    }

    void set_data_layout(homogen_data_layout dl) {
        layout_ = dl;
    }

    int64_t get_feature_count() const override {
        return feature_count_;
    }

    const table_feature& get_feature(int64_t feature_index) const override {
        return feature_;
    }

private:
    table_feature feature_;
    homogen_data_layout layout_;
    int64_t feature_count_;
};

} // namespace detail

table_feature::table_feature()
    : table_feature(data_type::float32) {}

table_feature::table_feature(data_type dtype)
    : table_feature(dtype,
                    is_floating_point(dtype) ? feature_type::ratio
                                             : feature_type::ordinal) {}

table_feature::table_feature(data_type dtype, feature_type ftype)
    : impl_(new detail::table_feature_impl{ dtype, ftype }) {}

data_type table_feature::get_data_type() const {
    return impl_->dtype;
}

table_feature& table_feature::set_data_type(data_type dt) {
    impl_->dtype = dt;
    return *this;
}

feature_type table_feature::get_type() const {
    return impl_->ftype;
}

table_feature& table_feature::set_type(feature_type ft) {
    impl_->ftype = ft;
    return *this;
}

table_metadata::table_metadata()
    : impl_(new detail::empty_metadata_impl()) {}

table_metadata::table_metadata(const table_feature& feature,
                               int64_t feature_count)
    : impl_(new detail::simple_metadata_impl {
        array<table_feature>(feature_count, feature)
    }) {}

table_metadata::table_metadata(array<table_feature> features)
    : impl_(new detail::simple_metadata_impl {
        features
    }) {}

int64_t table_metadata::get_feature_count() const {
    return impl_->get_feature_count();
}

const table_feature& table_metadata::get_feature(int64_t feature_index) const {
    return impl_->get_feature(feature_index);
}

using hm_impl = detail::homogen_table_metadata_impl;

homogen_table_metadata::homogen_table_metadata()
    : table_metadata(detail::pimpl<detail::table_metadata_impl>{
        new detail::homogen_table_metadata_impl()
    }) {}

homogen_table_metadata::homogen_table_metadata(const table_feature& feature,
                                               homogen_data_layout layout,
                                               int64_t feature_count)
    : table_metadata(detail::pimpl<detail::table_metadata_impl>{
        new detail::homogen_table_metadata_impl(feature, layout, feature_count)
    }) {}

homogen_data_layout homogen_table_metadata::get_data_layout() const {
    auto& impl = detail::get_impl<hm_impl>(*this);
    return impl.get_data_layout();
}

homogen_table_metadata& homogen_table_metadata::set_data_layout(homogen_data_layout dl) {
    auto& impl = detail::get_impl<hm_impl>(*this);
    impl.set_data_layout(dl);
    return *this;
}

} // namespace dal
