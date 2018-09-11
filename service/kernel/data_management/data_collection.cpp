/* file: data_collection.cpp */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

#include "data_management/data/data_collection.h"
#include "data_management/data/input_collection.h"

namespace daal
{
namespace data_management
{

namespace interface1
{

template<typename T> DAAL_EXPORT
services::SharedPtr<T> &KeyValueCollection<T>::operator[] (size_t k)
{
    size_t i;
    for( i = 0; i < _keys.size(); i++ )
    {
        if( _keys[i] == k )
        {
            return _values[i];
        }
    }
    _keys.push_back(k);
    _values.push_back( services::SharedPtr<T>() );
    return _values[i];
}

#define DAAL_INSTANTIATE_KEYVALUECOLLECTION(T)                                                \
    template DAAL_EXPORT services::SharedPtr<T> & KeyValueCollection<T>::operator[] (size_t k);


DAAL_INSTANTIATE_KEYVALUECOLLECTION(SerializationIface)
DAAL_INSTANTIATE_KEYVALUECOLLECTION(algorithms::Input)

/** @} */
} // namespace interface1

}
}
