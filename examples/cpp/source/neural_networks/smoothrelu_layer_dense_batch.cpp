/* file: smoothrelu_layer_dense_batch.cpp */
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
!    C++ example of forward and backward smooth rectified linear unit (smooth relu) layer usage
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SMOOTHRELU_LAYER_BATCH"></a>
 * \example smoothrelu_layer_dense_batch.cpp
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

int main()
{
    /* Read datasetFileName from a file and create a tensor to store input data */
    TensorPtr tensorData = readTensorFromCSV(datasetName);

    /* Create an algorithm to compute forward smooth relu layer results using default method */
    smoothrelu::forward::Batch<> smoothreluLayerForward;

    /* Set input objects for the forward smooth relu layer */
    smoothreluLayerForward.input.set(forward::data, tensorData);

    /* Compute forward smooth relu layer results */
    smoothreluLayerForward.compute();

    /* Print the results of the forward smooth relu layer */
    services::SharedPtr<smoothrelu::forward::Result> forwardResult = smoothreluLayerForward.getResult();
    printTensor(forwardResult->get(forward::value), "Forward smooth ReLU layer result (first 5 rows):", 5);

    /* Get the size of forward dropout smooth relu output */
    const Collection<size_t> &gDims = forwardResult->get(forward::value)->getDimensions();
    TensorPtr tensorDataBack = TensorPtr(new HomogenTensor<float>(gDims, Tensor::doAllocate, 0.01f));

    /* Create an algorithm to compute backward smooth relu layer results using default method */
    smoothrelu::backward::Batch<> smoothreluLayerBackward;

    /* Set input objects for the backward smooth relu layer */
    smoothreluLayerBackward.input.set(backward::inputGradient, tensorDataBack);
    smoothreluLayerBackward.input.set(backward::inputFromForward, forwardResult->get(forward::resultForBackward));

    /* Compute backward smooth relu layer results */
    smoothreluLayerBackward.compute();

    /* Print the results of the backward smooth relu layer */
    services::SharedPtr<backward::Result> backwardResult = smoothreluLayerBackward.getResult();
    printTensor(backwardResult->get(backward::gradient), "Backward smooth ReLU layer result (first 5 rows):", 5);

    return 0;
}
