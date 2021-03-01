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

#include "oneapi/dal/table/backend/interop/homogen_table_adapter.hpp"

namespace oneapi::dal::backend::interop {

namespace daal_dm = daal::data_management;

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

template <typename Data>
homogen_table_adapter<Data>::homogen_table_adapter(const homogen_table& table, status_t& status)
        // The following const_cast is safe only when this class is used for read-only
        // operations. Use on write leads to undefined behaviour.
        : base(daal_dm::DictionaryIface::equal,
               ptr_data_t{ const_cast<Data*>(table.get_data<Data>()), daal_object_owner(table) },
               table.get_column_count(),
               table.get_row_count(),
               status),
          original_table_(table) {
    if (!status.ok()) {
        return;
    }
    else if (!table.has_data()) {
        status.add(daal::services::ErrorIncorrectParameter);
        return;
    }

    this->_memStatus = daal_dm::NumericTableIface::userAllocated;
    this->_layout = daal_dm::NumericTableIface::aos;

    convert_feature_information_to_daal(original_table_.get_metadata(),
                                        *this->getDictionarySharedPtr());
}

template <typename Data>
auto homogen_table_adapter<Data>::releaseBlockOfRows(block_desc_t<double>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto homogen_table_adapter<Data>::releaseBlockOfRows(block_desc_t<float>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto homogen_table_adapter<Data>::releaseBlockOfRows(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto homogen_table_adapter<Data>::releaseBlockOfColumnValues(block_desc_t<double>& block)
    -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto homogen_table_adapter<Data>::releaseBlockOfColumnValues(block_desc_t<float>& block)
    -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto homogen_table_adapter<Data>::releaseBlockOfColumnValues(block_desc_t<int>& block) -> status_t {
    block.reset();
    return status_t();
}

template <typename Data>
auto homogen_table_adapter<Data>::assign(float) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto homogen_table_adapter<Data>::assign(double) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto homogen_table_adapter<Data>::assign(int) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto homogen_table_adapter<Data>::allocateDataMemoryImpl(daal::MemType) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto homogen_table_adapter<Data>::setNumberOfColumnsImpl(std::size_t) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
int homogen_table_adapter<Data>::getSerializationTag() const {
    ONEDAL_ASSERT(!"homogen_table_adapter: getSerializationTag() is not implemented");
    return -1;
}

template <typename Data>
auto homogen_table_adapter<Data>::serializeImpl(daal_dm::InputDataArchive* arch) -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
auto homogen_table_adapter<Data>::deserializeImpl(const daal_dm::OutputDataArchive* arch)
    -> status_t {
    return daal::services::ErrorMethodNotImplemented;
}

template <typename Data>
void homogen_table_adapter<Data>::freeDataMemoryImpl() {
    base::freeDataMemoryImpl();
    original_table_ = homogen_table{};
}

template class homogen_table_adapter<std::int32_t>;
template class homogen_table_adapter<float>;
template class homogen_table_adapter<double>;

} // namespace oneapi::dal::backend::interop
