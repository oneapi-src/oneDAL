/* file: DistributedPartialResultStep6.java */
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
 * @ingroup gbt_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.gbt.regression.Model;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP6"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of
 *        model-based training  in the sixth step of the distributed processing mode
 */
public final class DistributedPartialResultStep6 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of model-based training from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public DistributedPartialResultStep6(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep6();
    }

    DistributedPartialResultStep6(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns the partial result of model-based training
     * @param id    Identifier of the partial result
     * @return      Result that corresponds to the given identifier
     */
    public Model get(DistributedPartialResultStep6Id id) {
        int idValue = id.getValue();
        if (idValue != DistributedPartialResultStep6Id.partialModel.getValue()) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new Model(getContext(), cGetModel(cObject, idValue));
    }

    private native long cNewDistributedPartialResultStep6();

    private native long cGetModel(long resAddr, int id);
}
/** @} */
