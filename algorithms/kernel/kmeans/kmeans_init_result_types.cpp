/* file: kmeans_init_result_types.cpp */
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

#include "algorithms/kmeans/kmeans_init_types.h"
#include "daal_defines.h"
#include "serialization_utils.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace init
{
namespace interface1
{

__DAAL_REGISTER_SERIALIZATION_CLASS(Result, SERIALIZATION_KMEANS_INIT_RESULT_ID);

Result::Result() : daal::algorithms::Result(lastResultId + 1) {}

/**
 * Returns the result of computing initial clusters for the K-Means algorithm
 * \param[in] id   Identifier of the result
 * \return         Result that corresponds to the given identifier
 */
NumericTablePtr Result::get(ResultId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
 * Sets the result of computing initial clusters for the K-Means algorithm
 * \param[in] id    Identifier of the result
 * \param[in] ptr   Pointer to the object
 */
void Result::set(ResultId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Checks the result of computing initial clusters for the K-Means algorithm
* \param[in] input   %Input object for the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
services::Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
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

    const Parameter *kmPar = static_cast<const Parameter *>(par);
    const int unexpectedLayouts = (int)packed_mask;
    return checkNumericTable(get(centroids).get(), centroidsStr(), unexpectedLayouts, 0, nFeatures, kmPar->nClusters);
}

/**
* Checks the result of computing initial clusters for the K-Means algorithm
* \param[in] pres    Partial results of the algorithm
* \param[in] par     %Parameter of the algorithm
* \param[in] method  Computation method of the algorithm
*/
services::Status Result::check(const daal::algorithms::PartialResult *pres, const daal::algorithms::Parameter *par, int method) const
{
    size_t inputFeatures = const_cast<PartialResult *>(static_cast<const PartialResult *>(pres))->get(
                            partialClusters)->getNumberOfColumns();
    const Parameter *kmPar = static_cast<const Parameter *>(par);

    int unexpectedLayouts = (int)packed_mask;
    return checkNumericTable(get(centroids).get(), centroidsStr(), unexpectedLayouts, 0, inputFeatures, kmPar->nClusters);
}

} // namespace interface1
} // namespace init
} // namespace kmeans
} // namespace algorithm
} // namespace daal
