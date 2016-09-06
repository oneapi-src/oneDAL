/* file: concat_layer_dense_batch.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

/*
!  Content:
!    C++ example of forward and backward concatenation (concat) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-CONCAT_LAYER_BATCH"></a>
 * \example concat_layer_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::data_management;
using namespace daal::services;

/* Input data set parameters */
string datasetName = "../data/batch/layer.csv";
const size_t concatDimension = 1;
const size_t nInputs = 3;

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetName);

    /* Retrieve the input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);
    SharedPtr<LayerData> tensorDataCollection = SharedPtr<LayerData>(new LayerData());

    for(int i = 0; i < nInputs; i++)
    {
        (*tensorDataCollection)[i] = tensorData;
    }

    /* Create an algorithm to compute forward concatenation layer results using default method */
    concat::forward::Batch<> concatLayerForward(concatDimension);

    /* Set input objects for the forward concatenation layer */
    concatLayerForward.input.set(forward::inputLayerData, tensorDataCollection);

    /* Compute forward concatenation layer results */
    concatLayerForward.compute();

    /* Print the results of the forward concatenation layer */
    services::SharedPtr<concat::forward::Result> forwardResult = concatLayerForward.getResult();

    printTensor(forwardResult->get(forward::value), "Forward concatenation layer result value (first 5 rows):", 5);

    /* Create an algorithm to compute backward concatenation layer results using default method */
    concat::backward::Batch<> concatLayerBackward(concatDimension);

    /* Set inputs for the backward concatenation layer */
    concatLayerBackward.input.set(backward::inputGradient, forwardResult->get(forward::value));
    concatLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    printNumericTable(forwardResult->get(concat::auxInputDimensions), "auxInputDimensions ");

    /* Compute backward concatenation layer results */
    concatLayerBackward.compute();

    /* Print the results of the backward concatenation layer */
    services::SharedPtr<concat::backward::Result> backwardResult = concatLayerBackward.getResult();

    for(size_t i = 0; i < tensorDataCollection->size(); i++)
    {
        printTensor(backwardResult->get(backward::resultLayerData, i), "Backward concatenation layer backward result (first 5 rows):", 5);
    }

    return 0;
}
