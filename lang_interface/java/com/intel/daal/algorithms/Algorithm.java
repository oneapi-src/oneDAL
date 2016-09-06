/* file: Algorithm.java */
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

/**
 * @brief Contains classes that implement algorithms for data analysis (data mining), and data modeling (training and prediction).
 *        These algorithms include matrix decompositions, clustering algorithms, classification and regression algorithms,
 *        as well as association rules discovery.
 */

package com.intel.daal.algorithms;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__ALGORITHM"></a>
 * @brief Algorithm is the base class for the classes interfacing the major
 *        stages of data processing: Analysis, Training and Prediction.
 */
public abstract class Algorithm extends ContextClient {

    /**
     * @brief Pointer to C++ implementation of the Algorithm
     */
    public long cObject;

    /**
     * Constructs the algorithm
     * @param context  Context to manage the algorithm
     */
    public Algorithm(DaalContext context) {
        super(context);
    }

    public abstract void checkComputeParams();

    /**
     * Returns the newly allocated algorithm with a copy of input objects
     * and parameters of this algorithm
     * @return The newly allocated algorithm
     */
    @Override
    public abstract void dispose();

    public abstract Algorithm clone(DaalContext context);
}
