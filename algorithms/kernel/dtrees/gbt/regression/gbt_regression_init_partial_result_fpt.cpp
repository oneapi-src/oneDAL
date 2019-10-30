/* file: gbt_regression_init_partial_result_fpt.cpp */
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
//  Implementation of gbt regression classes.
//--
*/

#include "gbt_regression_init_types.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace regression
{
namespace init
{
namespace interface1
{

using namespace daal::data_management;

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep1::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);
    const DistributedInput<step1Local> *algInput = static_cast<const DistributedInput<step1Local> *>(input);

    const size_t maxBins = par->maxBins;
    const size_t nFeatures = algInput->get(step1LocalData)->getNumberOfColumns();

    services::Status status;
    set(step1MeanDependentVariable, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    set(step1NumberOfRows, HomogenNumericTable<size_t>::create(1, 1, NumericTable::doAllocate, &status));
    set(step1BinBorders, HomogenNumericTable<algorithmFPType>::create(nFeatures, maxBins + 1, NumericTable::doAllocate, &status));
    set(step1BinSizes, HomogenNumericTable<size_t>::create(nFeatures, maxBins, NumericTable::doAllocate, &status));
    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep2::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);
    const DistributedInput<step2Master> *algInput = static_cast<const DistributedInput<step2Master> *>(input);

    const size_t maxBins = par->maxBins;
    const size_t nFeatures = NumericTable::cast((*(algInput->get(step2BinSizes)))[0])->getNumberOfColumns();
    const size_t nNodes = algInput->get(step2NumberOfRows)->size();
    services::Status status;
    set(step2MergedBinBorders, HomogenNumericTable<algorithmFPType>::create(nFeatures, maxBins, NumericTable::doAllocate, &status));
    set(step2BinQuantities, HomogenNumericTable<size_t>::create(nFeatures, 1, NumericTable::doAllocate, &status));
    set(step2InitialResponse, HomogenNumericTable<algorithmFPType>::create(1, 1, NumericTable::doAllocate, &status));
    set(step2BinValues, DataCollectionPtr(new DataCollection()));
    return status;
}

/**
 * Allocates memory to store the results of gradient boosted trees model-based training
 * \param[in] input     Pointer to the structure of the input objects
 * \param[in] parameter Pointer to the structure of the algorithm parameters
 * \param[in] method    Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status DistributedPartialResultStep3::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *par = static_cast<const Parameter *>(parameter);
    const DistributedInput<step3Local> *algInput = static_cast<const DistributedInput<step3Local> *>(input);

    const size_t maxBins = par->maxBins;
    const size_t nRows = algInput->get(step3LocalData)->getNumberOfRows();
    const size_t nFeatures = algInput->get(step3LocalData)->getNumberOfColumns();

    services::Status status;
    // need fixes in java interfaces to work
     if (maxBins <= 256) {
         set(step3TransposedBinnedData, HomogenNumericTable<unsigned char>::create(nRows, nFeatures, NumericTable::doAllocate, &status));
         set(step3BinnedData, HomogenNumericTable<unsigned char>::create(nFeatures, nRows, NumericTable::doAllocate, &status));
     } else if (maxBins <= 65536) {
         set(step3TransposedBinnedData, HomogenNumericTable<unsigned short int>::create(nRows, nFeatures, NumericTable::doAllocate, &status));
         set(step3BinnedData, HomogenNumericTable<unsigned short int>::create(nFeatures, nRows, NumericTable::doAllocate, &status));
     } else {
         set(step3TransposedBinnedData, HomogenNumericTable<int>::create(nRows, nFeatures, NumericTable::doAllocate, &status));
         set(step3BinnedData, HomogenNumericTable<int>::create(nFeatures, nRows, NumericTable::doAllocate, &status));
     }

    set(step3Response, HomogenNumericTable<algorithmFPType>::create(1, nRows, NumericTable::doAllocate, &status));
    set(step3TreeOrder, HomogenNumericTable<int>::create(1, nRows, NumericTable::doAllocate, &status));

    return status;
}

template DAAL_EXPORT Status DistributedPartialResultStep1::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep2::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);
template DAAL_EXPORT Status DistributedPartialResultStep3::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

} // namespace interface1
} // namespace init
} // namespace regression
} // namespace gbt
} // namespace algorithms
} // namespace daal
