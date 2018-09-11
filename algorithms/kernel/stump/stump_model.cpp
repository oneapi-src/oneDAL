/* file: stump_model.cpp */
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
//  Implementation of the decision stump model constructor.
//--
*/

#include "algorithms/stump/stump_model.h"

namespace daal
{
namespace algorithms
{
namespace stump
{

/**
 * Empty constructor for deserialization
 */
Model::Model() : weak_learner::Model(), _nFeatures(0), _splitFeature(0), _values()
{}

size_t Model::getSplitFeature()
{
    return _splitFeature;
}

void Model::setSplitFeature(size_t splitFeature)
{
    _splitFeature = splitFeature;
}

}
}
} // namespace daal
