/* file: softmax_dense_batch.cpp */
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
!    C++ example of Softmax algorithm.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SOFTMAX_BATCH"></a>
 * \example softmax_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::math;

/* Input data set parameters */
string datasetName = "../data/batch/covcormoments_dense.csv";

int main()
{
    /* Retrieve the input data */
    FileDataSource<CSVFeatureManager> dataSource(datasetName, DataSource::doAllocateNumericTable, DataSource::doDictionaryFromContext);
    dataSource.loadDataBlock();

    /* Create an algorithm */
    softmax::Batch<float> softmax;

    /* Set an input object for the algorithm */
    softmax.input.set(softmax::data, dataSource.getNumericTable());

    /* Compute Softmax function */
    softmax.compute();

    /* Print the results of the algorithm */
    services::SharedPtr<softmax::Result> res = softmax.getResult();
    printNumericTable(res->get(softmax::value), "Softmax result (first 5 rows):", 5);

    return 0;
}
