/* file: out_detect_uni_dense_batch.cpp */
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
!    C++ example of univariate outlier detection
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-OUTLIER_DETECTION_UNIVARIATE_BATCH"></a>
 * \example out_detect_uni_dense_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace algorithms;

typedef double inputDataFPType;     /* Input data floating-point type */

/* Input data set parameters */
string datasetFileName = "../data/batch/outlierdetection.csv";

struct UserInitialization : public univariate_outlier_detection::InitIface
{
    size_t nFeatures;

    explicit UserInitialization(size_t nFeatures) : nFeatures(nFeatures) {}

    virtual void operator()(NumericTable *data,
                            NumericTable *location,
                            NumericTable *scatter,
                            NumericTable *threshold)
    {

        BlockDescriptor<double> locationBlock;
        BlockDescriptor<double> scatterBlock;
        BlockDescriptor<double> thresholdBlock;

        location->getBlockOfRows(0, 1, writeOnly, locationBlock);
        scatter->getBlockOfRows(0, 1, writeOnly, scatterBlock);
        threshold->getBlockOfRows(0, 1, writeOnly, thresholdBlock);

        for(size_t i = 0; i < nFeatures; i++)
        {
            locationBlock.getBlockPtr()[i]  = 0.0;
            scatterBlock.getBlockPtr()[i]   = 1.0;
            thresholdBlock.getBlockPtr()[i] = 3.0;
        }

        location->releaseBlockOfRows(locationBlock);
        scatter->releaseBlockOfRows(scatterBlock);
        threshold->releaseBlockOfRows(thresholdBlock);
    }
};

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    size_t nFeatures = dataSource.getNumberOfColumns();

    univariate_outlier_detection::Batch<> algorithm;

    algorithm.input.set(univariate_outlier_detection::data, dataSource.getNumericTable());

    algorithm.parameter.initializationProcedure = services::SharedPtr<univariate_outlier_detection::InitIface>(new UserInitialization(
                                                                                                                   nFeatures));

    /* Compute outliers */
    algorithm.compute();

    /* Get the computed results */
    services::SharedPtr<univariate_outlier_detection::Result> res = algorithm.getResult();

    printNumericTable(dataSource.getNumericTable(), "Input data");
    printNumericTable(res->get(univariate_outlier_detection::weights), "Outlier detection result (univariate)");

    return 0;
}
