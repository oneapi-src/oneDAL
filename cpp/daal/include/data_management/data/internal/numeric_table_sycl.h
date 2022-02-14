/* file: numeric_table_sycl.h */
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

#ifndef __SYCL_NUMERIC_TABLE_H__
#define __SYCL_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"

namespace daal
{
namespace data_management
{
namespace internal
{
namespace interface1
{
/**
 * @ingroup sycl
 * @{
 */

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__SYCLNUMERICTABLE"></a>
 *  \brief Base class for all numeric tables designed to work with SYCL* runtime.
 *  These tables avoid unnecessary data transfer between devices.
 */
class DAAL_EXPORT SyclNumericTable : public NumericTable
{
public:
    DAAL_CAST_OPERATOR(SyclNumericTable)

protected:
    explicit SyclNumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual, services::Status & st)
        : NumericTable(nColumns, nRows, featuresEqual, st)
    {}

    explicit SyclNumericTable(size_t nColumns, size_t nRows, DictionaryIface::FeaturesEqual featuresEqual)
        : NumericTable(nColumns, nRows, featuresEqual)
    {}

    explicit SyclNumericTable(NumericTableDictionaryPtr ddict, services::Status & st) : NumericTable(ddict, st) {}

    virtual ~SyclNumericTable() {}
};
typedef services::SharedPtr<SyclNumericTable> SyclNumericTablePtr;
typedef services::SharedPtr<const SyclNumericTable> SyclNumericTableConstPtr;

/** @} */

} // namespace interface1

using interface1::SyclNumericTable;
using interface1::SyclNumericTablePtr;
using interface1::SyclNumericTableConstPtr;

} // namespace internal
} // namespace data_management
} // namespace daal

#endif
