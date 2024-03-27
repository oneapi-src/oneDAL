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

#include <daal/include/data_management/data/data_dictionary.h>

#include "oneapi/dal/table/common.hpp"

#include "oneapi/dal/backend/interop/common.hpp"
#include "oneapi/dal/backend/interop/error_converter.hpp"
#include "oneapi/dal/backend/interop/data_conversion.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

daal_dm::features::FeatureType get_daal_feature_type(feature_type t) {
    switch (t) {
        case feature_type::nominal: return daal_dm::features::DAAL_CATEGORICAL;
        case feature_type::ordinal: return daal_dm::features::DAAL_ORDINAL;
        case feature_type::interval: return daal_dm::features::DAAL_CONTINUOUS;
        case feature_type::ratio: return daal_dm::features::DAAL_CONTINUOUS;
        default: throw dal::internal_error(detail::error_messages::unsupported_feature_type());
    }
}

feature_type get_dal_feature_type(daal_dm::features::FeatureType t) {
    switch (t) {
        case daal_dm::features::DAAL_ORDINAL: return feature_type::ordinal;
        case daal_dm::features::DAAL_CONTINUOUS: return feature_type::ratio;
        case daal_dm::features::DAAL_CATEGORICAL: return feature_type::nominal;
        default: throw dal::internal_error(detail::error_messages::unsupported_feature_type());
    }
}

void convert_feature_information_to_daal(const table_metadata& src,
                                         daal_dm::NumericTableDictionary& dst) {
    const auto feature_count = src.get_feature_count();
    [[maybe_unused]] const std::size_t casted_count = //
        dal::detail::integral_cast<std::size_t>(feature_count);
    ONEDAL_ASSERT(casted_count == dst.getNumberOfFeatures());

    for (std::int64_t i = 0l; i < feature_count; ++i) {
        const auto dtype = src.get_data_type(i);
        const auto ftype = src.get_feature_type(i);
        dst[i].indexType = getIndexNumType(dtype);
        dst[i].featureType = get_daal_feature_type(ftype);
    }

    interop::status_to_exception(dst.checkDictionary());
}

void convert_feature_information_from_daal(daal_dm::NumericTableDictionary& src,
                                           table_metadata& dst) {
    using detail::integral_cast;
    const auto f_number = src.getNumberOfFeatures();
    const auto f_count = integral_cast<std::int64_t>(f_number);

    auto dtypes = dal::array<data_type>::empty(f_count);
    auto ftypes = dal::array<feature_type>::empty(f_count);

    data_type* const d_ptr = dtypes.get_mutable_data();
    feature_type* const f_ptr = ftypes.get_mutable_data();

    for (std::int64_t i = 0l; i < f_count; ++i) {
        const auto idx = integral_cast<std::size_t>(i);
        const daal_dm::NumericTableFeature& feature = src[idx];

        d_ptr[idx] = get_dal_data_type(feature.indexType);
        f_ptr[idx] = get_dal_feature_type(feature.featureType);
    }

    dst = table_metadata{ dtypes, ftypes };
}

daal_dm::NumericTableDictionaryPtr convert(const table_metadata& src) {
    using daal_dm::NumericTableDictionary;
    using features_equal = daal_dm::DictionaryIface::FeaturesEqual;

    daal::services::Status status;
    const auto not_equal = features_equal::notEqual;
    const std::int64_t f_count = src.get_feature_count();
    const auto c_count = detail::integral_cast<std::size_t>(f_count);
    auto dict_ptr = NumericTableDictionary::create(c_count, not_equal, &status);
    interop::status_to_exception(status);

    convert_feature_information_to_daal(src, *dict_ptr);

    return dict_ptr;
}

table_metadata convert(const daal_dm::NumericTableDictionaryPtr& src) {
    table_metadata result{};
    convert_feature_information_from_daal(*src, result);
    return result;
}

} // namespace oneapi::dal::backend::interop
