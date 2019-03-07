/* file: algorithm_quality_metric_set_types.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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

/*
//++
//  Interface for the quality metric set.
//--
*/

#include "algorithm_quality_metric_set_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace quality_metric_set
{
namespace interface1
{

InputAlgorithmsCollection::InputAlgorithmsCollection(size_t n) : _qualityMetrics(n), _keys(n)
{
    nullPtr = new services::SharedPtr<quality_metric::Batch>();
}

InputAlgorithmsCollection::~InputAlgorithmsCollection()
{
    delete nullPtr;
}

/**
 * Returns a reference to SharedPtr for a stored object with a given key if an object with such key is registered
 * \param[in] k     Key value
 * \return Reference to SharedPtr of the quality_metric::Batch type
 */
const services::SharedPtr<quality_metric::Batch>& InputAlgorithmsCollection::operator[](size_t k) const
{
    size_t i;
    for (i = 0; i < _keys.size(); i++)
    {
        if (_keys[i] == k)
        {
            return _qualityMetrics[i];
        }
    }
    return *nullPtr;
}

/**
 * Returns a reference to SharedPtr for a stored object with a given key if an object with such key is registered.
 * Otherwise, creates an empty SharedPtr and stores it under the requested key and returns a reference for this value
 * \param[in] k     Key value
 * \return Reference to SharedPtr of the quality_metric::Batch type
 */
services::SharedPtr<quality_metric::Batch>& InputAlgorithmsCollection::operator[](size_t k)
{
    size_t i;
    for (i = 0; i < _keys.size(); i++)
    {
        if (_keys[i] == k)
        {
            return _qualityMetrics[i];
        }
    }
    _keys.push_back(0);
    _keys[i] = k;
    _qualityMetrics.push_back(services::SharedPtr<quality_metric::Batch>());
    return _qualityMetrics[i];
}

/**
 *  Returns the number of stored elements
 *  \return number of stored elements
 */
size_t InputAlgorithmsCollection::size() const { return _qualityMetrics.size(); }

/**
 * Removes all elements from the container
 */
void InputAlgorithmsCollection::clear()
{
    _keys.clear();
    _qualityMetrics.clear();
}

/**
 *  Returns a reference to SharedPtr for the stored key with a given index
 *  \param[in]  idx  Index of the requested key
 *  \return Reference to SharedPtr of the size_t type
 */
size_t InputAlgorithmsCollection::getKeyByIndex(int idx)
{
    return _keys[idx];
}



InputDataCollection::InputDataCollection() : data_management::KeyValueInputCollection() {}

/**
 * Adds an element with a key to the collection
 * \param[in] k     Key value
 * \param[in] ptr   Shared pointer to the element
 */
void InputDataCollection::add(size_t k, const algorithms::InputPtr& ptr)
{
    (*this)[k] = ptr;
}

/**
 * Returns the element that matches the key
 * \param[in] key     Key value
 * \return Shared pointer to the element
 */
algorithms::InputPtr InputDataCollection::getInput(size_t key) const
{
    return (*this)[key];
}


ResultCollection::ResultCollection() : data_management::KeyValueDataCollection() {}

void ResultCollection::add(size_t key, const algorithms::ResultPtr& ptr)
{
    (*this)[key] = ptr;
}

algorithms::ResultPtr ResultCollection::getResult(size_t key) const
{
    return services::staticPointerCast<algorithms::Result, data_management::SerializationIface>(this->operator[](key));
}


} // namespace interface1
} // namespace quality_metric_set
} // namespace algorithms
} // namespace daal
