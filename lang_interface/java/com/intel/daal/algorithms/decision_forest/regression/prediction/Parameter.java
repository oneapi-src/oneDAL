/* file: Parameter.java */
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
 * @ingroup decision_forest_regression_prediction
 */
/**
 * @brief Contains parameter for decision forest regression prediction algorithm
 */
package com.intel.daal.algorithms.decision_forest.regression.prediction;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DECISION_FOREST__REGRESSION__PREDICTION__PARAMETER"></a>
 * @brief Parameter of the decision forest regression prediction algorithm
 */
public class Parameter extends com.intel.daal.algorithms.Parameter {

    public Parameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }
}
/** @} */
