/* file: DistributedStep6LocalInput.java */
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
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__DISTRIBUTEDSTEP6LOCALINPUT"></a>
 * @brief %Input objects for the DBSCAN algorithm in the sixth step of the distributed processing mode
 */

public final class DistributedStep6LocalInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public DistributedStep6LocalInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the DBSCAN algorithm in the sixth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(LocalCollectionInputId id, DataCollection val) {
        if (id == LocalCollectionInputId.partialData || id == LocalCollectionInputId.partialWeights) {
            cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Adds an input object for the DBSCAN algorithm in the sixth step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param val           Value of the input object
     */
    public void add(LocalCollectionInputId id, NumericTable val) {
        if (id == LocalCollectionInputId.partialData || id == LocalCollectionInputId.partialWeights) {
            cAddNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the DBSCAN algorithm in the sixth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public DataCollection get(LocalCollectionInputId id) {
        if (id == LocalCollectionInputId.partialData || id == LocalCollectionInputId.partialWeights) {
            return (DataCollection)Factory.instance().createObject(getContext(), cGetDataCollection(getCObject(), id.getValue()));
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets an input object for the DBSCAN algorithm in the sixth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(Step6LocalCollectionInputId id, DataCollection val) {
        if (id == Step6LocalCollectionInputId.haloData || id == Step6LocalCollectionInputId.haloDataIndices ||
            id == Step6LocalCollectionInputId.haloWeights || id == Step6LocalCollectionInputId.haloBlocks) {
            cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Adds an input object for the DBSCAN algorithm in the sixth step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param val           Value of the input object
     */
    public void add(Step6LocalCollectionInputId id, NumericTable val) {
        if (id == Step6LocalCollectionInputId.haloData || id == Step6LocalCollectionInputId.haloDataIndices ||
            id == Step6LocalCollectionInputId.haloWeights || id == Step6LocalCollectionInputId.haloBlocks) {
            cAddNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the DBSCAN algorithm in the sixth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public DataCollection get(Step6LocalCollectionInputId id) {
        if (id == Step6LocalCollectionInputId.haloData || id == Step6LocalCollectionInputId.haloDataIndices ||
            id == Step6LocalCollectionInputId.haloWeights || id == Step6LocalCollectionInputId.haloBlocks) {
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
