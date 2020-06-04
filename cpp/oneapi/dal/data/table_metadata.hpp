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

#pragma once

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/data/array.hpp"

namespace dal {

enum class data_layout {
    row_major,
    column_major
};

enum class feature_type {
    nominal,
    ordinal,
    contiguous
};

struct feature_info {
    feature_info()
        : dtype(data_type::float32),
          ftype(feature_type::contiguous) { }

    feature_info(data_type dtype)
        : dtype(dtype) {
        if (dtype == data_type::float32 || dtype == data_type::float64) {
            ftype = feature_type::contiguous;
        } else {
            ftype = feature_type::nominal;
        }
    }

    feature_info(feature_type ftype)
        : ftype(ftype) {
        if (ftype == feature_type::nominal || ftype == feature_type::ordinal) {
            dtype = data_type::int32;
        } else {
            dtype = data_type::float32;
        }
    }

    feature_info(data_type dtype, feature_type ftype)
        : dtype(dtype),
          ftype(ftype) { }

    data_type dtype;
    feature_type ftype;
};

struct table_metadata {
    table_metadata()
        : layout(data_layout::row_major) {}

    table_metadata(std::int64_t features_count,
                   feature_info feature = {},
                   data_layout layout = data_layout::row_major)
        : layout(layout),
          features(features_count, feature) {}

    data_layout layout;
    array<feature_info> features;
};

} // namespace dal
