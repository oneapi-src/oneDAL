/* file: kmeans_init_result.h */
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
//  Implementation of kmeans classes.
//--
*/

#ifndef __KMEANS_INIT_RESULT_
#define __KMEANS_INIT_RESULT_

#include "algorithms/kmeans/kmeans_init_types.h"

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{

/**
 * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
 * \param[in] input        Pointer to the input structure
 * \param[in] parameter    Pointer to the parameter structure
 * \param[in] method       Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *kmPar = static_cast<const Parameter *>(parameter);
    const DistributedStep2MasterInput* masterInput = dynamic_cast<const DistributedStep2MasterInput*>(input);
    size_t nFeatures = 0;
    if(masterInput)
    {
        data_management::DataCollection* coll = masterInput->get(partialResults).get();
        for(size_t i = 0; i < coll->size(); ++i)
        {
            data_management::NumericTable* partClusters = static_cast<PartialResult *>((*coll)[i].get())->get(partialClusters).get();
            if(partClusters)
            {
                nFeatures = partClusters->getNumberOfColumns();
                break;
            }
        }
    }
    else
        nFeatures = (static_cast<const Input *>(input))->get(data)->getNumberOfColumns();

    Argument::set(centroids, data_management::SerializationIfacePtr(
        new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, kmPar->nClusters, data_management::NumericTable::doAllocate)));
    return services::Status();
}

/**
 * Allocates memory to store the results of computing initial clusters for the K-Means algorithm
 * \param[in] partialResult Pointer to the partial result structure
 * \param[in] parameter     Pointer to the parameter structure
 * \param[in] method        Computation method of the algorithm
 */
template <typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::PartialResult *partialResult, const daal::algorithms::Parameter *parameter, const int method)
{
    const Parameter *kmPar = static_cast<const Parameter *>(parameter);

    size_t nClusters = kmPar->nClusters;
    size_t nFeatures = 0;
    const DistributedStep5MasterPlusPlusPartialResult* step5 = dynamic_cast<const DistributedStep5MasterPlusPlusPartialResult*>(partialResult);
    if(step5)
        nFeatures = step5->get(candidates)->getNumberOfColumns();
    else
    {
    const data_management::NumericTable* pPartialClusters = static_cast<const PartialResult *>(partialResult)->get(partialClusters).get();
    if(pPartialClusters)
        nFeatures = pPartialClusters->getNumberOfColumns();
    }
    Argument::set(centroids, data_management::SerializationIfacePtr(
                      new data_management::HomogenNumericTable<algorithmFPType>(nFeatures, nClusters,
                                                                                data_management::NumericTable::doAllocate)));
    return services::Status();
}

} // namespace init
} // namespace kmeans
} // namespace algorithms
} // namespace daal

#endif
