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

#include <daal/include/data_management/data/data_dictionary.h>
#include "oneapi/dal/table/homogen.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

template <typename Body>
static daal::services::Status convert_exception_to_status(Body&& body) {
    try {
        return body();
    }
    catch (const bad_alloc&) {
        return daal::services::ErrorMemoryAllocationFailed;
    }
    catch (const out_of_range&) {
        return daal::services::ErrorIncorrectDataRange;
    }
    catch (...) {
        return daal::services::UnknownError;
    }
}

static daal_dm::features::FeatureType get_daal_feature_type(feature_type t) {
    switch (t) {
        case feature_type::nominal: return daal_dm::features::DAAL_CATEGORICAL;
        case feature_type::ordinal: return daal_dm::features::DAAL_ORDINAL;
        case feature_type::interval: return daal_dm::features::DAAL_CONTINUOUS;
        case feature_type::ratio: return daal_dm::features::DAAL_CONTINUOUS;
        default: throw dal::internal_error(detail::error_messages::unsupported_feature_type());
    }
}

static void convert_feature_information_to_daal(const table_metadata& src,
                                                daal_dm::NumericTableDictionary& dst) {
    ONEDAL_ASSERT(std::size_t(src.get_feature_count()) == dst.getNumberOfFeatures());
    for (std::int64_t i = 0; i < src.get_feature_count(); i++) {
        auto& daal_feature = dst[i];
        daal_feature.featureType = get_daal_feature_type(src.get_feature_type(i));
    }
}

} // namespace oneapi::dal::backend::interop
