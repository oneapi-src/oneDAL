/* file: BernoulliDenseBatch.java */
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

/*
 //  Content:
 //     Java example of bernoulli distribution
 ////////////////////////////////////////////////////////////////////////////////
 */

package com.intel.daal.examples.distributions;

import com.intel.daal.algorithms.distributions.*;
import com.intel.daal.algorithms.distributions.bernoulli.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.examples.utils.Service;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-EXAMPLE-JAVA-BERNOULLIDENSEBATCH">
 * @example BernoulliDenseBatch.java
 */
class BernoulliDenseBatch {
    private static DaalContext context = new DaalContext();

    public static void main(String[] args) throws java.io.FileNotFoundException, java.io.IOException {
        /* Create input table to fill with random numbers */
        HomogenNumericTable dataTable = new HomogenNumericTable(context, Float.class, 1, 10, NumericTable.AllocationFlag.DoAllocate);

        /* Create the algorithm */
        Batch bernoulli = new Batch(context, Float.class, Method.defaultDense, 0.5);

        /* Set the algorithm input */
        bernoulli.input.set(InputId.tableToFill, dataTable);

        /* Set the Mersenne Twister engine to the distribution */
        com.intel.daal.algorithms.engines.mt19937.Batch eng = new com.intel.daal.algorithms.engines.mt19937.Batch(context, Float.class, com.intel.daal.algorithms.engines.mt19937.Method.defaultDense, 777);
        bernoulli.parameter.setEngine(eng);

        /* Perform computations */
        bernoulli.compute();

        /* Print the results */
        Service.printNumericTable("Bernoulli distribution output:", dataTable);

        context.dispose();
    }
}
