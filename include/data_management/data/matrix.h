/* file: matrix.h */
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
#include "data_management/data/data_serialize.h"

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
template<typename DataType = DAAL_DATA_TYPE>
class DAAL_EXPORT Matrix : public HomogenNumericTable<DataType>
{
public:
    DECLARE_SERIALIZABLE_TAG();

    /**
     *  Constructor for a matrix
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     */
    Matrix( size_t nColumns = 0, size_t nRows = 0, DataType *const ptr = 0 ):
        HomogenNumericTable<DataType>( services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows ) { }

    /**
     *  Constructs a matrix
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[out] stat        Status of the matrix construction
     *  \return Matrix
     */
    static services::SharedPtr<Matrix<DataType> > create(size_t nColumns = 0, size_t nRows = 0, DataType *const ptr = 0,
                                                         services::Status *stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), stat);
    }

    /**
     *  Constructor for a matrix
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \DAAL_DEPRECATED_USE{ Matrix::create }
     */
    Matrix( size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr):
        HomogenNumericTable<DataType>( ptr, nColumns, nRows ) { }

    /**
     *  Constructs a matrix
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[out] stat        Status of the matrix construction
     *  \return Matrix
     */
    static services::SharedPtr<Matrix<DataType> > create(size_t nColumns, size_t nRows, services::SharedPtr<DataType> &ptr,
                                                         services::Status *stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, ptr, stat);
    }

    /**
     *  Constructor for a matrix
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     */
    Matrix( DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns = 0, size_t nRows = 0, DataType *const ptr = 0 ):
        HomogenNumericTable<DataType>( featuresEqual, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows ) { }

    /**
     *  Constructs a matrix
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[out] stat        Status of the matrix construction
     *  \return Matrix
     */
    static services::SharedPtr<Matrix<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                         size_t nColumns = 0, size_t nRows = 0, DataType *const ptr = 0,
                                                         services::Status *stat = NULL)
    {
        return create(featuresEqual, nColumns, nRows, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), stat);
    }

    /**
     *  Constructor for a matrix
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     */
    Matrix( DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr ):
        HomogenNumericTable<DataType>( featuresEqual, ptr, nColumns, nRows ) { }

    /**
     *  Constructs a matrix
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[out] stat        Status of the matrix construction
     *  \return Matrix
     */
    static services::SharedPtr<Matrix<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                         size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr,
                                                         services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(Matrix, DataType, featuresEqual, nColumns, nRows, ptr);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     */
    Matrix( size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag ):
        HomogenNumericTable<DataType>( nColumns, nRows, memoryAllocationFlag ) { }

    /**
     *  Constructs a matrix with memory allocation controlled via a flag
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                    Status of the matrix construction
     *  \return Matrix
     */
    static services::SharedPtr<Matrix<DataType> > create(size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag,
                                                         services::Status *stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, memoryAllocationFlag, stat);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag
     *  \param[in]  featuresEqual           Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     */
    Matrix( DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag ):
        HomogenNumericTable<DataType>( featuresEqual, nColumns, nRows, memoryAllocationFlag ) { }

    /**
     *  Constructs a matrix with memory allocation controlled via a flag
     *  \param[in]  featuresEqual           Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[out] stat                    Status of the matrix construction
     *  \return Matrix
     */
    static services::SharedPtr<Matrix<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                         size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag,
                                                         services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(Matrix, DataType, featuresEqual, nColumns, nRows, memoryAllocationFlag);
    }

    /**
     *  Constructor for a matrix. Fills the table with a constant
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue  Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( size_t nColumns, size_t nRows, DataType *const ptr, const DataType &constValue ):
        HomogenNumericTable<DataType>( services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows, constValue ) { }

    /**
     *  Constructs a matrix and fills it with a constant
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the matrix construction
     *  \return     Matrix initialized with a constant
     */
    static services::SharedPtr<Matrix<DataType> > create(size_t nColumns, size_t nRows, DataType *const ptr,
                                                         const DataType &constValue, services::Status *stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), constValue, stat);
    }

    /**
     *  Constructor for a matrix. Fills the table with a constant
     *  \param[in]  nColumns    Number of columns in the table
     *  \param[in]  nRows       Number of rows in the table
     *  \param[in]  ptr         Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue  Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr, const DataType &constValue ):
        HomogenNumericTable<DataType>( ptr, nColumns, nRows, constValue ) { }

    /**
     *  Constructs a matrix and fills it with a constant
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the matrix construction
     *  \return     Matrix initialized with a constant
     */
    static services::SharedPtr<Matrix<DataType> > create(size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr,
                                                         const DataType &constValue, services::Status *stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, ptr, constValue, stat);
    }

    /**
     *  Constructor for a matrix. Fills the table with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, DataType *const ptr, const DataType &constValue ):
        HomogenNumericTable<DataType>( featuresEqual, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), nColumns, nRows, constValue ) { }

    /**
     *  Constructs a matrix and fills it with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the matrix construction
     *  \return     Matrix initialized with a constant
     */
    static services::SharedPtr<Matrix<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                         size_t nColumns, size_t nRows, DataType *const ptr,
                                                         const DataType &constValue, services::Status *stat = NULL)
    {
        return create(featuresEqual, nColumns, nRows, services::SharedPtr<DataType>(ptr, services::EmptyDeleter()), constValue, stat);
    }

    /**
     *  Constructor for a matrix. Fills the table with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr, const DataType &constValue ):
        HomogenNumericTable<DataType>( featuresEqual, ptr, nColumns, nRows, constValue ) { }

    /**
     *  Constructs a matrix and fills it with a constant
     *  \param[in]  featuresEqual  Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns       Number of columns in the table
     *  \param[in]  nRows          Number of rows in the table
     *  \param[in]  ptr            Pointer to and an array with a homogeneous data set
     *  \param[in]  constValue     Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat           Status of the matrix construction
     *  \return     Matrix initialized with a constant
     */
    static services::SharedPtr<Matrix<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                         size_t nColumns, size_t nRows, const services::SharedPtr<DataType> &ptr,
                                                         const DataType &constValue, services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(Matrix, DataType, featuresEqual, nColumns, nRows, ptr, constValue);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag,
            const DataType &constValue ):
        HomogenNumericTable<DataType>( nColumns, nRows, memoryAllocationFlag, constValue ) { }

    /**
     *  Constructor for a matrix with memory allocation controlled via a flag and filling the matrix with a constant
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat                    Status of the matrix construction
     *  \return Matrix initialized with a constant
     */
    static services::SharedPtr<Matrix<DataType> > create(size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag,
                                                         const DataType &constValue, services::Status *stat = NULL)
    {
        return create(DictionaryIface::notEqual, nColumns, nRows, memoryAllocationFlag, constValue, stat);
    }

    /**
     *  Constructor for a Numeric Table with memory allocation controlled via a flag and filling the table with a constant
     *  \param[in]  featuresEqual           Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     */
    Matrix( DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
            NumericTable::AllocationFlag memoryAllocationFlag, const DataType &constValue ):
        HomogenNumericTable<DataType>( featuresEqual, nColumns, nRows, memoryAllocationFlag, constValue ) { }

    /**
     *  Constructor for a matrix with memory allocation controlled via a flag and filling the matrix with a constant
     *  \param[in]  featuresEqual           Flag that makes all features in the Numeric Table Data Dictionary equal
     *  \param[in]  nColumns                Number of columns in the table
     *  \param[in]  nRows                   Number of rows in the table
     *  \param[in]  memoryAllocationFlag    Flag that controls internal memory allocation for data in the numeric table
     *  \param[in]  constValue              Constant to initialize entries of the homogeneous numeric table
     *  \param[out] stat                    Status of the matrix construction
     *  \return Matrix initialized with a constant
     */
    static services::SharedPtr<Matrix<DataType> > create(DictionaryIface::FeaturesEqual featuresEqual,
                                                         size_t nColumns, size_t nRows, NumericTable::AllocationFlag memoryAllocationFlag,
                                                         const DataType &constValue, services::Status *stat = NULL)
    {
        DAAL_DEFAULT_CREATE_TEMPLATE_IMPL_EX(Matrix, DataType, featuresEqual, nColumns, nRows, memoryAllocationFlag, constValue);
    }

    /** \private */
    virtual ~Matrix() { }

protected:

    Matrix(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
           const services::SharedPtr<DataType> &ptr, services::Status &st):
        HomogenNumericTable<DataType>(featuresEqual, ptr, nColumns, nRows, st) { }

    Matrix(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
           NumericTable::AllocationFlag memoryAllocationFlag, services::Status &st):
        HomogenNumericTable<DataType>(featuresEqual, nColumns, nRows, memoryAllocationFlag, st) { }

    Matrix(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
           const services::SharedPtr<DataType> &ptr, const DataType &constValue,
           services::Status &st):
        HomogenNumericTable<DataType>(featuresEqual, ptr, nColumns, nRows, constValue, st) { }

    Matrix(DictionaryIface::FeaturesEqual featuresEqual, size_t nColumns, size_t nRows,
           NumericTable::AllocationFlag memoryAllocationFlag, const DataType &constValue, services::Status &st):
        HomogenNumericTable<DataType>(featuresEqual, nColumns, nRows, memoryAllocationFlag, constValue, st) { }
};
/** @} */
} // namespace interface1
using interface1::Matrix;

}
} // namespace daal
#endif
