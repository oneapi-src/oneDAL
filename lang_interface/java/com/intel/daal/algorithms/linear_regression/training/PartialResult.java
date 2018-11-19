/* file: PartialResult.java */
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
 * @ingroup linear_regression_training
 * @{
 */
package com.intel.daal.algorithms.linear_regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.ComputeStep;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.linear_regression.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__LINEAR_REGRESSION__TRAINING__PARTIALRESULT"></a>
 * @brief Provides methods to access a partial result obtained with the compute() method of linear regression
 *        model-based training in the online or distributed processing mode
 */
public final class PartialResult extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public PartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of linear regression model-based training
     * @param id    Identifier of the partial result
     * @return      Partial result that corresponds to the given identifier
     */
    public Model get(PartialResultId id) {
        if (id == PartialResultId.model) {
            return new Model(getContext(), cGetModel(getCObject(), PartialResultId.model.getValue()));
        } else {
            return null;
        }
    }

    private native long cGetModel(long resAddr, int id);
}
/** @} */
