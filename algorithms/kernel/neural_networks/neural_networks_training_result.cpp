/* file: neural_networks_training_result.cpp */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
