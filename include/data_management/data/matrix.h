/* file: matrix.h */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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
//  Implementation of a matrix numeric table.
//--
*/

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "services/daal_memory.h"
#include "data_management/data/numeric_table.h"
#include "data_management/data/homogen_numeric_table.h"
#include "services/daal_shared_ptr.h"

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
 *  <a name="DAAL-CLASS-DATA_MANAGEMENT__MATRIX"></a>
 *  \brief Represents a two-dimensional table of numbers of the same type
 *  \tparam DataType Defines the underlying data type that describes the matrix
 */
template<typename DataType = double>
class Matrix : public HomogenNumericTable<DataType>
{
public:
    /**
     *  Constructor for a matrix
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     */
    Matrix( size_t nColumns = 0, size_t nRows = 0, DataType *const ptr = 0 ):
        HomogenNumericTable<DataType>( ptr, nColumns, nRows ) { }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     */
    Matrix( size_t nColumns,   size_t nRows,   NumericTable::AllocationFlag memoryAllocationFlag ):
        HomogenNumericTable<DataType>( nColumns, nRows, memoryAllocationFlag ) { }

    /**
     *  Constructor for a matrix. Fills the table with a constant
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue  Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( size_t nColumns,   size_t nRows,   DataType *const ptr, const DataType &constValue ):
        HomogenNumericTable<DataType>( ptr, nColumns, nRows, constValue ) { }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( size_t nColumns,   size_t nRows,   NumericTable::AllocationFlag memoryAllocationFlag,
            const DataType &constValue ):
        HomogenNumericTable<DataType>( nColumns, nRows, memoryAllocationFlag, constValue ) { }

    /** \private */
    virtual ~Matrix() { }

    virtual int getSerializationTag() DAAL_C11_OVERRIDE
    {
        return data_feature_utils::getIndexNumType<DataType>() + SERIALIZATION_MATRIX_NT_ID;
    }

};
/** @} */
} // namespace interface1
using interface1::Matrix;

}
} // namespace daal
#endif
