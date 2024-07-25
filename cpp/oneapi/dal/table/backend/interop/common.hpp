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

#pragma once

#include <daal/include/data_management/data/data_dictionary.h>
#include <daal/include/data_management/data/csr_numeric_table.h>
#include <daal/include/data_management/data/soa_numeric_table.h>
#include <daal/include/data_management/data/homogen_numeric_table.h>

#include "oneapi/dal/table/common.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

template <typename Type>
using shared_t = daal::services::SharedPtr<Type>;

using soa_table_t = daal_dm::SOANumericTable;
using csr_table_t = daal_dm::CSRNumericTable;
template <typename Type>
using homogen_table_t = daal_dm::CSRNumericTable;

using soa_table_ptr_t = shared_t<soa_table_t>;
using csr_table_ptr_t = shared_t<csr_table_t>;
template <typename Type>
using homogen_table_ptr_t = shared_t<homogen_table_t<Type>>;

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

ONEDAL_EXPORT daal_dm::features::FeatureType get_daal_feature_type(feature_type t);

ONEDAL_EXPORT feature_type get_daal_feature_type(daal_dm::features::FeatureType t);

ONEDAL_EXPORT void convert_feature_information_to_daal(const table_metadata& src,
                                                       daal_dm::NumericTableDictionary& dst);

ONEDAL_EXPORT void convert_feature_information_from_daal(const daal_dm::NumericTableDictionary& src,
                                                         table_metadata& dst);

ONEDAL_EXPORT daal_dm::NumericTableDictionaryPtr convert(const table_metadata& src);

ONEDAL_EXPORT table_metadata convert(const daal_dm::NumericTableDictionaryPtr& src);

} // namespace oneapi::dal::backend::interop
