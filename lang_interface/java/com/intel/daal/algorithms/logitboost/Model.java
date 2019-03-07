/* file: Model.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation.
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
 * @defgroup logitboost Logitboost Classifier
 * @brief Contains classes for the LogitBoost classification algorithm
 * @ingroup boosting
 */
package com.intel.daal.algorithms.logitboost;

import com.intel.daal.utils.*;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LOGITBOOST__MODEL"></a>
 * @brief %Model of the classifier trained by the LogitBoost algorithm in the batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.boosting.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
        _nIterations = cGetIterations(cModel);
    }

    /**
     * Returns the number of iterations done by the training algorithm
     * @return The number of iterations done by the training algorithm
     */
    public long getIterations() {
        return _nIterations;
    }

    private long _nIterations;

    private native long cGetIterations(long modelAddr);
}
/** @} */
