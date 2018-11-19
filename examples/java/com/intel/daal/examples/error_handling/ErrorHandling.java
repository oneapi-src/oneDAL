/* file: ErrorHandling.java */
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
 //     Java example of the error handling
 ////////////////////////////////////////////////////////////////////////////////
 */

/**
 * <a name="DAAL-EXAMPLE-JAVA-ERRORHANDLING">
 * @example ErrorHandling.java
 */

package com.intel.daal.examples.error_handling;

import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.services.DaalContext;

class ErrorHandling {
    /* Input data set parameters */
    private static final String wrongDataset = "../data/batch/wrongName.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Retrieve the input data */
        try {
            FileDataSource dataSource = new FileDataSource(context, wrongDataset,
                                                           DataSource.DictionaryCreationFlag.DoDictionaryFromContext,
                                                           DataSource.NumericTableAllocationFlag.DoAllocateNumericTable);
        }
        catch(Exception e) {
            System.out.println("FileDataSource expected error: " + e.getMessage());
        }
    }
}
