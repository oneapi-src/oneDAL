/* file: base_arrow_numeric_table.h */
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

#ifndef __BASE_ARROW_NUMERIC_TABLE_H__
#define __BASE_ARROW_NUMERIC_TABLE_H__

#include "data_management/data/numeric_table.h"
#include "data_management/data/internal/conversion.h"

namespace daal
{
namespace data_management
{
namespace interface1
{
/**
 * @ingroup numeric_tables
 * @{
 */
/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__ARROWNUMERICTABLE"></a>
 *  \brief Base class that provides methods to access data stored as a Apache Arrow table.
 */
class DAAL_EXPORT ArrowNumericTable : public NumericTable
{
public:
    /**
     *  Returns whether the numeric table is mutable
     *  \return true if the numeric table is mutable, false otherwise
     */
    virtual bool isMutable() const = 0;

protected:
    ArrowNumericTable(size_t featnum, size_t obsnum, services::Status & st)
        : NumericTable(featnum, obsnum, DictionaryIface::notEqual, st) {}
};
typedef services::SharedPtr<ArrowNumericTable> ArrowNumericTablePtr;

/**
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__BASEARROWIMMUTABLENUMERICTABLE"></a>
 *  \brief Base class that provides methods to access data stored as a immutable Apache Arrow table.
 */
class DAAL_EXPORT BaseArrowImmutableNumericTable : public ArrowNumericTable
{
    DECLARE_SERIALIZABLE_TAG();

public:
    bool isMutable() const DAAL_C11_OVERRIDE { return false; }

protected:
    BaseArrowImmutableNumericTable(size_t featnum, size_t obsnum, services::Status & st) : ArrowNumericTable(featnum, obsnum, st) {}
};
typedef services::SharedPtr<BaseArrowImmutableNumericTable> BaseArrowImmutableNumericTablePtr;

/** @} */
} // namespace interface1

using interface1::ArrowNumericTable;
using interface1::ArrowNumericTablePtr;
using interface1::BaseArrowImmutableNumericTable;
using interface1::BaseArrowImmutableNumericTablePtr;

} // namespace data_management
} // namespace daal

#endif
