/* file: gbt_training_parameter.cpp */
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
//  Implementation of gradient boosted trees training parameter class
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_training_parameter.h"
#include "daal_strings.h"
#include "gbt_internal.h"
#include "algorithms/engines/mt19937/mt19937.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace training
{
using namespace daal::services;

Parameter::Parameter() : splitMethod(defaultSplit), maxIterations(50), maxTreeDepth(6),
    shrinkage(0.3), minSplitLoss(0.), lambda(1.),
    observationsPerTreeFraction(1.),
    featuresPerNode(0),
    minObservationsInLeafNode(5),
    memorySavingMode(false),
    engine(engines::mt19937::Batch<>::create()),
    minBinSize(5),
    maxBins(256),
    internalOptions(gbt::internal::parallelAll)
{
}

Status checkImpl(const gbt::training::Parameter& prm)
{
    DAAL_CHECK_EX(prm.maxIterations, ErrorIncorrectParameter, ParameterName, maxIterationsStr());
    DAAL_CHECK_EX((prm.shrinkage > 0) && (prm.shrinkage <= 1), ErrorIncorrectParameter, ParameterName, shrinkageStr());
    DAAL_CHECK_EX((prm.minSplitLoss >= 0), ErrorIncorrectParameter, ParameterName, minSplitLossStr());
    DAAL_CHECK_EX((prm.lambda >= 0), ErrorIncorrectParameter, ParameterName, lambdaStr());
    DAAL_CHECK_EX((prm.observationsPerTreeFraction > 0) && (prm.observationsPerTreeFraction <= 1),
        ErrorIncorrectParameter, ParameterName, observationsPerTreeFractionStr());
    DAAL_CHECK_EX(prm.minObservationsInLeafNode, ErrorIncorrectParameter, ParameterName, minObservationsInLeafNodeStr());
    if(prm.splitMethod == inexact)
    {
        DAAL_CHECK_EX((prm.maxBins >= 2), ErrorIncorrectParameter, ParameterName, maxBinsStr());
        DAAL_CHECK_EX((prm.minBinSize >= 1), ErrorIncorrectParameter, ParameterName, minBinSizeStr());
    }
    return Status();
}

} // namespace training
} // namespace gbt
} // namespace algorithms
} // namespace daal
