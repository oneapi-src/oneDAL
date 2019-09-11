/* file: qr_dense_default_distr_step2.h */
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
//  Implementation of qr algorithm and types methods.
//--
*/
#ifndef __QR_DENSE_DEFAULT_DISTR_STEP2__
#define __QR_DENSE_DEFAULT_DISTR_STEP2__

#include "qr_types.h"

using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace qr
{
namespace interface1
{

/**
 * Allocates memory for storing partial results of the QR decomposition algorithm
 * \param[in] input  Pointer to input object
 * \param[in] parameter    Pointer to parameter
 * \param[in] method Computation method
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResult::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    Argument::set(outputOfStep2ForStep3, data_management::KeyValueDataCollectionPtr(new data_management::KeyValueDataCollection()));
    Argument::set(finalResultFromStep2Master, ResultPtr(new Result()));
    data_management::KeyValueDataCollectionPtr inCollection = static_cast<const DistributedStep2Input *>(input)->get(inputOfStep2FromStep1);
    size_t nBlocks = 0;
    return setPartialResultStorage<algorithmFPType>(inCollection.get(), nBlocks);
}

/**
 * Allocates memory for storing partial results of the QR decomposition algorithm based on known structure of partial results from the
 * first steps of the algorithm in the distributed processing mode.
 * KeyValueDataCollection under outputOfStep2ForStep3 is structured the same as KeyValueDataCollection under
 * inputOfStep2FromStep1 id of the algorithm input
 * \tparam     algorithmFPType             Data type to be used for storage in resulting HomogenNumericTable
 * \param[in]  inCollection  KeyValueDataCollection of all partial results from the first steps of the algorithm in the distributed
 * processing mode
 * \param[out] nBlocks  Number of rows in the input data set
 */
template <typename algorithmFPType>
DAAL_EXPORT Status DistributedPartialResult::setPartialResultStorage(data_management::KeyValueDataCollection *inCollection, size_t &nBlocks)
{
    data_management::KeyValueDataCollectionPtr partialCollection =
        services::staticPointerCast<data_management::KeyValueDataCollection, data_management::SerializationIface>(Argument::get(outputOfStep2ForStep3));
    if(!partialCollection)
    {
        return Status();
    }

    ResultPtr result = services::staticPointerCast<Result, data_management::SerializationIface>(Argument::get(finalResultFromStep2Master));

    size_t inSize = inCollection->size();

    data_management::DataCollection *fisrtNodeCollection = static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex(0).get());
    data_management::NumericTable   *firstNumericTable   = static_cast<data_management::NumericTable *>((*fisrtNodeCollection)[0].get());

    size_t m = firstNumericTable->getNumberOfColumns();
    if(result->get(matrixR).get() == NULL)
    {
        result->allocateImpl<algorithmFPType>(m, 0);
    }

    nBlocks = 0;
    Status s;
    for(size_t i = 0 ; i < inSize ; i++)
    {
        data_management::DataCollection *nodeCollection = static_cast<data_management::DataCollection *>((*inCollection).getValueByIndex((int)i).get());
        size_t nodeKey  = (*inCollection).getKeyByIndex((int)i);
        size_t nodeSize = nodeCollection->size();
        nBlocks += nodeSize;

        data_management::DataCollectionPtr nodePartialResult(new data_management::DataCollection());

        for(size_t j = 0 ; j < nodeSize ; j++)
        {
            nodePartialResult->push_back(data_management::HomogenNumericTable<algorithmFPType>::create(m, m, data_management::NumericTable::doAllocate, &s));
        }
        (*partialCollection)[ nodeKey ] = nodePartialResult;
    }
    return s;
}

}// namespace interface1
}// namespace qr
}// namespace algorithms
}// namespace daal

#endif
