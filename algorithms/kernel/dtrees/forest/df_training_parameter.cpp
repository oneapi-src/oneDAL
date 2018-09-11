/* file: df_training_parameter.cpp */
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
//  Implementation of decision forest training parameter class
//--
*/

#include "algorithms/decision_forest/decision_forest_training_parameter.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace decision_forest
{
namespace training
{
using namespace daal::services;
Status checkImpl(const decision_forest::training::Parameter& prm)
{
    DAAL_CHECK_EX(prm.nTrees, ErrorIncorrectParameter, ParameterName, nTreesStr());
    DAAL_CHECK_EX(prm.minObservationsInLeafNode, ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    DAAL_CHECK_EX((prm.observationsPerTreeFraction > 0) && (prm.observationsPerTreeFraction <= 1),
        ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());
    DAAL_CHECK_EX((prm.impurityThreshold >= 0), ErrorIncorrectParameter, ParameterName, impurityThresholdStr());
    Status s;
    if(!prm.bootstrap)
    {
        if(prm.varImportance == MDA_Raw || prm.varImportance == MDA_Scaled)
            s.add(Error::create(ErrorDFBootstrapVarImportanceIncompatible));
        if(prm.resultsToCompute&computeOutOfBagError)
            s.add(Error::create(ErrorDFBootstrapOOBIncompatible));
    }
    return s;
}

} // namespace training
} // namespace decision_forest
} // namespace algorithms
} // namespace daal
