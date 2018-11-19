/* file: AbsCSRBatch.java */
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
 //     Java example of Abs algorithm
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.math;

import com.intel.daal.algorithms.math.abs.*;
import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data_source.DataSource;
import com.intel.daal.data_management.data_source.FileDataSource;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-ABSCSRBATCH">
 * @example AbsCSRBatch.java
 */

class AbsCSRBatch {
    private static final String dataset = "../data/batch/covcormoments_csr.csv";

    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Read a data set from a file and create a numeric table for storing the input data */
        CSRNumericTable input = Service.createSparseTable(context, dataset);

        /* Create an algorithm */
        Batch absAlgorithm = new Batch(context, Float.class, Method.fastCSR);

        /* Set an input object for the algorithm */
        absAlgorithm.input.set(InputId.data, input);

        /* Compute Abs function */
        Result result = absAlgorithm.compute();

        /* Print the results of the algorithm */
        Service.printNumericTable("Abs result (first 5 rows):", (CSRNumericTable)result.get(ResultId.value), 5);

        context.dispose();
    }
}
