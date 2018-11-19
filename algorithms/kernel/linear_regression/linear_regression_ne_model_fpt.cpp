/* file: linear_regression_ne_model_fpt.cpp */
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
//  Implementation of the class defining the linear regression model
//--
*/

#include "linear_regression_ne_model_impl.h"
#include "data_management/data/homogen_numeric_table.h"

namespace daal
{
namespace algorithms
{
namespace linear_regression
{
namespace internal
{
using namespace daal::data_management;

/**
 * Constructs the linear regression model for the normal equations method
 * \param[in] featnum Number of features in the training data set
 * \param[in] nrhs    Number of responses in the training data
 * \param[in] par     Parameters of linear regression model-based training
 * \param[in] dummy   Dummy variable for the templated constructor
 */
template <typename modelFPType>
ModelNormEqInternal::ModelNormEqInternal(size_t featnum, size_t nrhs, const linear_regression::Parameter &par, modelFPType dummy, Status &st):
    super(featnum, nrhs, par, dummy)
{
    size_t dimWithoutBeta = getNumberOfBetas();
    if(!_interceptFlag)
    {
        dimWithoutBeta--;
    }
    _xtxTable = HomogenNumericTable<modelFPType>::create(dimWithoutBeta, dimWithoutBeta, NumericTable::doAllocate, 0, &st);
    if (!st) return;
    _xtyTable = HomogenNumericTable<modelFPType>::create(dimWithoutBeta, nrhs, NumericTable::doAllocate, 0, &st);
    if (!st) return;
}

template ModelNormEqInternal::ModelNormEqInternal(size_t featnum, size_t nrhs, const linear_regression::Parameter &par, DAAL_FPTYPE dummy, Status &st);
}// namespace internal
}// namespace linear_regression
}// namespace algorithms
}// namespace daal
