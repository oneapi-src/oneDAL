/* file: common_helpers.h */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
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

#include "common_helpers_batch.h"
#include "common_helpers_distributed.h"
#include "common_helpers_online.h"
#include "common_helpers_argument.h"
#include "common_helpers_functions.h"

#define USING_COMMON_NAMESPACES()          \
    using namespace daal;                  \
    using namespace daal::data_management; \
    using namespace daal::algorithms;      \
    using namespace daal::services;

namespace daal
{
template <typename Object>
inline jlong pack(Object * object)
{
    return reinterpret_cast<jlong>(object);
}

template <typename Object>
inline Object & unpack(jlong objectAddress)
{
    return *reinterpret_cast<Object *>(objectAddress);
}

template <typename Implementation, typename Abstract>
inline jlong packAbstractPtr(const daal::services::SharedPtr<Implementation> & object)
{
    auto abstractObject = new daal::services::SharedPtr<Abstract>();
    *abstractObject     = daal::services::staticPointerCast<Abstract, Implementation>(object);
    return pack<daal::services::SharedPtr<Abstract> >(abstractObject);
}

template <typename Implementation, typename Abstract>
inline daal::services::SharedPtr<Implementation> unpackAbstractPtr(jlong objectAddress)
{
    auto abstractObject = unpack<daal::services::SharedPtr<Abstract> >(objectAddress);
    return daal::services::staticPointerCast<Implementation, Abstract>(abstractObject);
}

template <typename Algorithm>
inline jlong packAlgorithm(const daal::services::SharedPtr<Algorithm> & object)
{
    return packAbstractPtr<Algorithm, daal::algorithms::AlgorithmIface>(object);
}

template <typename Algorithm>
inline daal::services::SharedPtr<Algorithm> unpackAlgorithm(jlong objectAddress)
{
    return unpackAbstractPtr<Algorithm, daal::algorithms::AlgorithmIface>(objectAddress);
}

template <typename Model>
inline jlong packModel(const daal::services::SharedPtr<Model> & object)
{
    return packAbstractPtr<Model, daal::data_management::SerializationIface>(object);
}

template <typename Model>
inline daal::services::SharedPtr<Model> unpackModel(jlong objectAddress)
{
    return unpackAbstractPtr<Model, daal::data_management::SerializationIface>(objectAddress);
}

template <typename Table>
inline jlong packTable(const daal::services::SharedPtr<Table> & object)
{
    return packAbstractPtr<Table, daal::data_management::SerializationIface>(object);
}

template <typename Table>
inline daal::services::SharedPtr<Table> unpackTable(jlong objectAddress)
{
    return unpackAbstractPtr<Table, daal::data_management::SerializationIface>(objectAddress);
}

} // namespace daal
