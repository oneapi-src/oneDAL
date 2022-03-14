/* file: data_source_utils.h */
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

/*
//++
//  Declaration and implementation of the base data source class.
//--
*/

#ifndef __DATA_SOURCE_UTILS_H__
#define __DATA_SOURCE_UTILS_H__

#include "data_management/data_source/data_source_dictionary.h"
#include "data_management/data/numeric_table.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup data_sources
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__STRINGROWFEATUREMANAGERIFACE"></a>
 *  \brief Abstract interface class that defines the interface to parse and convert the raw data represented as a string into a numeric format.
 *         The string must represent a row of data, a dictionary, or a vector of features
 */
class StringRowFeatureManagerIface
{
public:
    virtual ~StringRowFeatureManagerIface() {}

    /**
     *  Parses a string that represents features of a data set and constructs a dictionary
     *  \param[in]  rawRowData   Array of characters with a string that contains information about features of the data set
     *  \param[in]  rawDataSize  Size of the rawRowData array
     *  \param[out] dict         Pointer to the dictionary constructed from the string
     */
    virtual void parseRowAsDictionary(char * rawRowData, size_t rawDataSize, DataSourceDictionary * dict) = 0;

    /**
     *  Parses a string that represents a feature vector and converts it into a numeric representation
     *  \param[in]  rawRowData   Array of characters with a string that represents the feature vector
     *  \param[in]  rawDataSize  Size of the rawRowData array
     *  \param[in]  dict         Pointer to the dictionary
     *  \param[out] rowBuffer    Pointer to a Buffer View to store the result of parsing
     *  \param[in]  ntRowIndex   Position in the Numeric Table at which to store the result of parsing
     */
    virtual void parseRowIn(char * rawRowData, size_t rawDataSize, DataSourceDictionary * dict, services::BufferView<DAAL_DATA_TYPE> & rowBuffer,
                            size_t ntRowIndex) = 0;
};
/** @} */
} // namespace interface1
using interface1::StringRowFeatureManagerIface;

} // namespace data_management
} // namespace daal

#endif
