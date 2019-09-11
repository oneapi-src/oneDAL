/* file: DistributedPartialResultStep4.java */
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
 * @ingroup implicit_als_training_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP4"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        implicit ALS training algorithm in the fourth step of the distributed processing mode
 */
public final class DistributedPartialResultStep4 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of the implicit ALS training algorithm from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public DistributedPartialResultStep4(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep4();
    }

    public DistributedPartialResultStep4(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep4Id
     * @return Partial result that corresponds to the given identifier
     */
    public PartialModel get(DistributedPartialResultStep4Id id) {
        int idValue = id.getValue();
        if (id != DistributedPartialResultStep4Id.outputOfStep4ForStep1 &&
            id != DistributedPartialResultStep4Id.outputOfStep4ForStep3 &&
            id != DistributedPartialResultStep4Id.outputOfStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new PartialModel(getContext(), cGetPartialModel(getCObject(), idValue));
    }

    /**
    * Sets a partial result of the implicit ALS training algorithm obtained in the fourth step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedPartialResultStep4Id id, PartialModel value) {
        int idValue = id.getValue();
        if (id != DistributedPartialResultStep4Id.outputOfStep4ForStep1 &&
            id != DistributedPartialResultStep4Id.outputOfStep4ForStep3 &&
            id != DistributedPartialResultStep4Id.outputOfStep4) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialModel(getCObject(), idValue, value.getCObject());
    }

    private native long cNewDistributedPartialResultStep4();

    private native long cGetPartialModel(long cObject, int id);
    private native void cSetPartialModel(long cObject, int id, long cPartialModel);
}
/** @} */
