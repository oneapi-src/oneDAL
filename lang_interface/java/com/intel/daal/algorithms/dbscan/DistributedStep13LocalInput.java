/* file: DistributedStep13LocalInput.java */
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
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDSTEP13LOCALINPUT"></a>
 * @brief %Input objects for the DBSCAN algorithm in the thirteenth step of the distributed processing mode
 */

public final class DistributedStep13LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep13LocalInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the DBSCAN algorithm in the thirteenth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step13LocalCollectionInputId id, DataCollection val) {
        if (id == Step13LocalCollectionInputId.partialAssignmentQueries) {
            cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Adds an input object for the DBSCAN algorithm in the thirteenth step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param val           Value of the input object
     */
    public void add(Step13LocalCollectionInputId id, NumericTable val) {
        if (id == Step13LocalCollectionInputId.partialAssignmentQueries) {
            cAddNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the DBSCAN algorithm in the thirteenth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public DataCollection get(Step13LocalCollectionInputId id) {
        if (id == Step13LocalCollectionInputId.partialAssignmentQueries) {
            return (DataCollection)Factory.instance().createObject(getContext(), cGetDataCollection(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    private native void cSetDataCollection(long cObject, int id, long dcAddr);
    private native void cAddNumericTable(long cObject, int id, long ntAddr);
    private native long cGetDataCollection(long cObject, int id);
}
/** @} */
