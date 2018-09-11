/* file: TrainingResult.java */
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
 * @ingroup implicit_als_training
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__TRAININGRESULT"></a>
 * @brief Provides methods to access the results of the implicit ALS training algorithm
 */
public final class TrainingResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result for the implicit ALS training algorithm
     * @param context Context to manage the partial result for the implicit ALS training algorithm
     */
    public TrainingResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public TrainingResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of the implicit ALS training algorithm
     * @param  id   Identifier of the result
     * @return      Result that corresponds to the given identifier
     */
    public Model get(TrainingResultId id) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("TrainingResultId unsupported");
        }
        return new Model(getContext(), cGetResultModel(getCObject(), id.getValue()));
    }

    /**
     * Sets the result of the implicit ALS training algorithm
     * @param id    Identifier of the result
     * @param value Result that corresponds to the given identifier
     */
    public void set(TrainingResultId id, Model value) {
        if (id != TrainingResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultModel(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultModel(long resAddr, int id);
    private native void cSetResultModel(long resAddr, int id, long modelAddr);
}
/** @} */
