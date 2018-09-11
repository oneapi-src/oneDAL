/* file: TrainingPartialResult.java */
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
 * @ingroup training
 * @{
 */
package com.intel.daal.algorithms.classifier.training;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__CLASSIFIER__TRAINING__TRAININGPARTIALRESULT"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the classifier model training algorithm in the online
 *        or distributed processing mode
 */

public class TrainingPartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result for the classifier training algorithm in the online or distributed processing mode
     * @param context   Context to manage the partial result for the classifier training algorithm in the online or distributed processing mode
     */
    public TrainingPartialResult(DaalContext context) {
        super(context);
    }

    public TrainingPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }
}
/** @} */
