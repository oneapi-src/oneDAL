/* file: datasource_featureextraction.cpp */
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
!  Content:
!    C++ example for using of data source feature extraction
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASOURCE_FEATUREEXTRACTION"></a>
 * \example datasource_featureextraction.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

/* Input data set parameters */
string datasetFileName     = "../data/batch/kmeans_dense.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable);

    /* Create data source dictionary from loading of the first .csv file */
    dataSource.createDictionaryFromContext();

    /* Filter in 3 chosen columns from a .csv file */
    services::Collection<size_t> validList(3);
    validList[0] = 1;
    validList[1] = 2;
    validList[2] = 5;

    dataSource.getFeatureManager().addModifier( ColumnFilter().list(validList) );

    /* Consider column with index 1 as categorical and convert it into 3 binary categorical features */
    dataSource.getFeatureManager().addModifier( OneHotEncoder(1, 3) );

    /* Load data from .csv file */
    dataSource.loadDataBlock();

    /* Print result */
    NumericTablePtr table = dataSource.getNumericTable();
    printNumericTable(table, "Loaded data", 4, 20);

    return 0;
}
