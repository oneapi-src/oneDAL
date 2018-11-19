/* file: kmeans_input_types.cpp */
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

#include "algorithms/kmeans/kmeans_types.h"
#include "daal_defines.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace kmeans
{
namespace interface1
{

Input::Input() : InputIface(lastInputId + 1) {}

/**
* Returns an input object for the K-Means algorithm
* \param[in] id    Identifier of the input object
* \return          %Input object that corresponds to the given identifier
*/
NumericTablePtr Input::get(InputId id) const
{
    return staticPointerCast<NumericTable, SerializationIface>(Argument::get(id));
}

/**
* Sets an input object for the K-Means algorithm
* \param[in] id    Identifier of the input object
* \param[in] ptr   Pointer to the object
*/
void Input::set(InputId id, const NumericTablePtr &ptr)
{
    Argument::set(id, ptr);
}

/**
* Returns the number of features in the input object
* \return Number of features in the input object
*/
size_t Input::getNumberOfFeatures() const
{
    NumericTablePtr inTable = get(data);
    return inTable->getNumberOfColumns();
}

/**
* Checks input objects for the K-Means algorithm
* \param[in] par     Algorithm parameter
* \param[in] method  Computation method of the algorithm
*/
services::Status Input::check(const daal::algorithms::Parameter *parameter, int method) const
{
    services::Status s;
    const Parameter *kmPar = static_cast<const Parameter *>(parameter);
    const int expectedLayout = (method == lloydCSR ? (int)NumericTableIface::csrArray : 0);
    DAAL_CHECK_STATUS(s, checkNumericTable(get(data).get(), dataStr(), 0, expectedLayout));
    const size_t inputFeatures = get(data)->getNumberOfColumns();
    const size_t inputRows = get(data)->getNumberOfRows();
    if (kmPar->maxIterations > 0)
    {
        DAAL_CHECK(inputRows >= kmPar->nClusters, ErrorKMeansNumberOfClustersIsTooLarge);
    }
    return checkNumericTable(get(inputCentroids).get(), inputCentroidsStr(), 0, 0, inputFeatures, kmPar->nClusters);
}

} // namespace interface1
} // namespace kmeans
} // namespace algorithm
} // namespace daal
