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
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.dbscan;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDPARTIALRESULTSTEP5"></a>
 * @brief Provides methods to access partial results obtained with the compute() method of the
 *        DBSCAN algorithm in the fifth step of the distributed processing mode
 */
public final class DistributedPartialResultStep5 extends com.intel.daal.algorithms.PartialResult {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs a partial result of the DBSCAN algorithm from the context
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
     * Returns a partial result of the DBSCAN algorithm obtained in the fifth step of the distributed processing mode
     * @param  id   Identifier of the input object, @ref DistributedPartialResultStep5Id
     * @return      Partial result that corresponds to the given identifier
     */
    public DataCollection get(DistributedPartialResultStep5Id id) {
        if (id != DistributedPartialResultStep5Id.partitionedHaloData && id != DistributedPartialResultStep5Id.partitionedHaloDataIndices &&
            id != DistributedPartialResultStep5Id.partitionedHaloWeights) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (DataCollection)Factory.instance().createObject(getContext(), cGetDataCollection(getCObject(), id.getValue()));
    }

    /**
    * Sets a partial result of the DBSCAN algorithm obtained in the fifth step of the distributed processing mode
    * @param id     Identifier of the input object
    * @param value  Value of the input object
    */
    public void set(DistributedPartialResultStep5Id id, DataCollection value) {
        if (id != DistributedPartialResultStep5Id.partitionedHaloData && id != DistributedPartialResultStep5Id.partitionedHaloDataIndices &&
            id != DistributedPartialResultStep5Id.partitionedHaloWeights) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetDataCollection(getCObject(), id.getValue(), value.getCObject());
    }

    private native long cNewDistributedPartialResultStep5();

    private native long cGetDataCollection(long cObject, int id);
    private native void cSetDataCollection(long cObject, int id, long cDataCollection);
}
/** @} */
