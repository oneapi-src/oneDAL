/* file: DistributedPartialResultStep3.java */
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
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP3"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of
 *        model-based training  in the third step of the distributed processing mode
 */
public final class DistributedPartialResultStep3 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of model-based training from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public DistributedPartialResultStep3(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep3();
    }

    DistributedPartialResultStep3(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of the model-based training obtained in the third step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep3Id
     * @return      Partial result that corresponds to the given identifier
     */
    public DataCollection get(DistributedPartialResultStep3Id id) {
        if (id != DistributedPartialResultStep3Id.histograms) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (DataCollection)Factory.instance().createObject(getContext(), cGetDataCollection(getCObject(), id.getValue()));
    }

    /**
    * Sets a partial result of the model-based training obtained in the third step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedPartialResultStep3Id id, DataCollection value) {
        if (id != DistributedPartialResultStep3Id.histograms) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetDataCollection(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewDistributedPartialResultStep3();

    private native long cGetDataCollection(long cObject, int id);
    private native void cSetDataCollection(long cObject, int id, long cDataCollection);
}
/** @} */
