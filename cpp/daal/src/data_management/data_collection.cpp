/* file: data_collection.cpp */
/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "data_management/data/data_collection.h"
#include "data_management/data/input_collection.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
template <typename T>
services::SharedPtr<T> & KeyValueCollection<T>::operator[](size_t k)
{
    size_t i;
    for (i = 0; i < _keys.size(); i++)
    {
        if (_keys[i] == k)
        {
            return _values[i];
        }
    }
    _keys.push_back(k);
    _values.push_back(services::SharedPtr<T>());
    return _values[i];
}

#define DAAL_INSTANTIATE_KEYVALUECOLLECTION(T) template services::SharedPtr<T> & KeyValueCollection<T>::operator[](size_t k);

DAAL_INSTANTIATE_KEYVALUECOLLECTION(SerializationIface)
DAAL_INSTANTIATE_KEYVALUECOLLECTION(algorithms::Input)

/** @} */
} // namespace interface1

} // namespace data_management
} // namespace daal
