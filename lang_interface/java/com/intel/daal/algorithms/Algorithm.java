/* file: Algorithm.java */
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

/**
 * @defgroup algorithms Algorithms
 * @{
 */
/**
 * @defgroup base_algorithms Base Classes
 * @ingroup algorithms
 * @{
 */
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
/** @} */
/** @} */
