/* file: InitResult.java */
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
 * @ingroup implicit_als_init
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITRESULT"></a>
 * @brief Provides methods to access the results of computing the initial model for the
 * implicit ALS training algorithm
 */
public final class InitResult extends com.intel.daal.algorithms.Result {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the result for the implicit ALS initialization algorithm in the distributed processing mode
     * @param context Context to manage the result for the implicit ALS initialization algorithm in the distributed processing mode
     */
    public InitResult(DaalContext context) {
        super(context);
        this.cObject = cNewResult();
    }

    public InitResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the result of computing the initial model for the implicit ALS training algorithm
     * @param id   Identifier of the result
     * @return         Result that corresponds to the given identifier
     */
    public Model get(InitResultId id) {
        if (id != InitResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetResultModel(cObject, id.getValue()));
    }

    /**
     * Sets the result of computing the initial model for the implicit ALS training algorithm
     * @param id    Identifier of the result
     * @param value Result that corresponds to the given identifier
     */
    public void set(InitResultId id, Model value) {
        int idValue = id.getValue();
        if (id != InitResultId.model) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetResultModel(cObject, idValue, value.getCObject());
    }

    private native long cNewResult();

    private native long cGetResultModel(long cResult, int id);

    private native void cSetResultModel(long cResult, int id, long cModel);
}
/** @} */
