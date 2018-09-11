/* file: neural_networks_training_result.cpp */
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

#include "neural_networks_training_result.h"
#include "daal_strings.h"

using namespace daal::data_management;
using namespace daal::services;

namespace daal
{
namespace algorithms
{
namespace neural_networks
{
namespace training
{

Result::Result() : daal::algorithms::Result(lastResultId + 1)
{
    set(model, Model::create());
}

ModelPtr Result::get(ResultId id) const
{
    return Model::cast(Argument::get(id));
}

void Result::set(ResultId id, const ModelPtr &value)
{
    Argument::set(id, value);
}

Status Result::check(const daal::algorithms::Input *input, const daal::algorithms::Parameter *par, int method) const
{
    ModelPtr trainModel = get(model);
    DAAL_CHECK(trainModel, ErrorNullModel)
    const Parameter *param = static_cast<const Parameter *>(par);
    if(param->optimizationSolver)
    {
        size_t batchSizeFromModel = 0;
        if(trainModel->getForwardLayer(0) && trainModel->getForwardLayer(0)->getLayerInput() && trainModel->getForwardLayer(0)->getLayerInput()->get(layers::forward::data))
        {
            batchSizeFromModel = trainModel->getForwardLayer(0)->getLayerInput()->get(layers::forward::data)->getDimensionSize(0);
        }
        DAAL_CHECK(batchSizeFromModel == param->optimizationSolver->getParameter()->batchSize, ErrorInconsistenceModelAndBatchSizeInParameter);
    }
    return Status();
}

}
}
}
}
