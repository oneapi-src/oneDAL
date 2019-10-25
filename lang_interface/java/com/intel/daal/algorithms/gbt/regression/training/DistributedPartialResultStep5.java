/* file: DistributedPartialResultStep5.java */
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
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__GBT__REGRESSION__TRAINING__DISTRIBUTEDPARTIALRESULTSTEP5"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of
 *        model-based training  in the fifth step of the distributed processing mode
 */
public final class DistributedPartialResultStep5 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of model-based training from the context
     * @param context Context to manage the memory in the native part of the partial result object
     */
    public DistributedPartialResultStep5(DaalContext context) {
        super(context);
        this.cObject = cNewDistributedPartialResultStep5();
    }

    DistributedPartialResultStep5(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of the model-based training obtained in the fifth step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep5Id
     * @return      Partial result that corresponds to the given identifier
     */
    public NumericTable get(DistributedPartialResultStep5Id id) {
        if (id != DistributedPartialResultStep5Id.step5TreeStructure && id != DistributedPartialResultStep5Id.step5TreeOrder) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetNumericTable(getCObject(), id.getValue()));
    }

    /**
    * Sets a partial result of the model-based training obtained in the fifth step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedPartialResultStep5Id id, NumericTable value) {
        if (id != DistributedPartialResultStep5Id.step5TreeStructure && id != DistributedPartialResultStep5Id.step5TreeOrder) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetNumericTable(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewDistributedPartialResultStep5();

    private native long cGetNumericTable(long cObject, int id);
    private native void cSetNumericTable(long cObject, int id, long cNumericTable);
}
/** @} */
