/* file: smoothrelu_dense_batch.cpp */
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
!    C++ example of SmoothReLU algorithm.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SMOOTHRELU_BATCH"></a>
 * \example smoothrelu_dense_batch.cpp
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
    smoothrelu::Batch<float> smoothReLU;

    /* Set an input object for the algorithm */
    smoothReLU.input.set(smoothrelu::data, dataSource.getNumericTable());

    /* Compute SmoothReLU function */
    smoothReLU.compute();

    /* Print the results of the algorithm */
    services::SharedPtr<smoothrelu::Result> res = smoothReLU.getResult();
    printNumericTable(res->get(smoothrelu::value), "SmoothReLU result (first 5 rows):", 5);

    return 0;
}
