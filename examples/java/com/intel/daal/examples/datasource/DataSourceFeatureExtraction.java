/* file: DataSourceFeatureExtraction.java */
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
 //  Content:
 //    Java example for using of data source feature extraction
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-DATASOURCEFEATUREEXTRACTION">
 * @example DataSourceFeatureExtraction.java
 */

package com.intel.daal.examples.datasource;

import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data_source.*;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

class DataSourceFeatureExtraction {

    /* Input data set parameters */
    private static final String dataset = "../data/batch/kmeans_dense.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {

        /* Retrieve the input data */
        FileDataSource dataSource = new FileDataSource(context, dataset,
                DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);

        /* Filter in 3 chosen columns from a .csv file */
        dataSource.getFeatureManager().addModifier(new ColumnFilter(context).list(new long[]{1, 2, 5}));

        /* Consider column with index 1 as categorical and convert it into 3 binary categorical features */
        dataSource.getFeatureManager().addModifier(new OneHotEncoder(context, 1, 3));

        /* Load data from .csv file */
        dataSource.loadDataBlock();
        NumericTable table = dataSource.getNumericTable();

        /* Print result */
        Service.printNumericTable("Loaded data", table, 4, 20);

        context.dispose();
    }
}
