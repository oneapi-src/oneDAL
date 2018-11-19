/* file: Model.java */
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
 * @defgroup brownboost Brownboost Classifier
 * @brief Contains classes for the BrownBoost classification algorithm
 * @ingroup boosting
 */
package com.intel.daal.algorithms.brownboost;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__BROWNBOOST__MODEL"></a>
 * @brief %Model of the classifier trained by the BrownBoost algorithm in the batch processing mode.
 */
public class Model extends com.intel.daal.algorithms.boosting.Model {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public Model(DaalContext context, long cModel) {
        super(context, cModel);
    }

    /**
     * Returns the numeric table that contains the array of weights of weak learners constructed
     * during training of the BrownBoost algorithm.
     * The size of the array equals the number of weak learners
     * @return Array of weights of weak learners.
     */
    public NumericTable getAlpha() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetAlpha(this.getCObject()));
    }

    private NumericTable _alpha;

    private native long cGetAlpha(long modelAddr);
}
/** @} */
