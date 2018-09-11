/* file: gbt_classification_training_result_fpt.cpp */
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
//  Implementation of the gradient boosted trees algorithm interface
//--
*/

#include "algorithms/gradient_boosted_trees/gbt_classification_training_types.h"
#include "gbt_classification_model_impl.h"

namespace daal
{
namespace algorithms
{
namespace gbt
{
namespace classification
{
namespace training
{

template<typename algorithmFPType>
DAAL_EXPORT services::Status Result::allocate(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method)
{
    services::Status s;
    const classifier::training::Input* inp = static_cast<const classifier::training::Input*>(input);
    const size_t nFeatures = inp->get(classifier::training::data)->getNumberOfColumns();
    set(classifier::training::model, daal::algorithms::gbt::classification::Model::create(nFeatures, &s));
    return s;
}

template DAAL_EXPORT services::Status Result::allocate<DAAL_FPTYPE>(const daal::algorithms::Input *input, const daal::algorithms::Parameter *parameter, const int method);

}// namespace training
}// namespace classification
}// namespace gbt
}// namespace algorithms
}// namespace daal
