/* file: InitDistributedStep2MasterInput.java */
/*******************************************************************************
* Copyright 2014-2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

/**
 * @ingroup dbscan_distributed
 * @{
 */
package com.intel.daal.algorithms.gbt.regression.init;

import com.intel.daal.utils.*;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.DataCollection;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__DBSCAN__InitDistributedStep2MasterInput"></a>
 * @brief %Input objects for the DBSCAN algorithm in the ninth step of the distributed processing mode
 */

public final class InitDistributedStep2MasterInput extends com.intel.daal.algorithms.Input {

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public InitDistributedStep2MasterInput(DaalContext context, long cObject) {
        super(context);
        this.cObject = cObject;
    }

    /**
     * Sets an input object for the DBSCAN algorithm in the ninth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @param val     Value of the input object
     */
    public void set(InitStep2MasterCollectionInputId id, DataCollection val) {
        if (id == InitStep2MasterCollectionInputId.step2MeanDependentVariable || id == InitStep2MasterCollectionInputId.step2NumberOfRows ||
            id == InitStep2MasterCollectionInputId.step2BinBorders || id == InitStep2MasterCollectionInputId.step2BinSizes) {
            cSetDataCollection(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Adds an input object for the DBSCAN algorithm in the ninth step of the distributed processing mode
     * @param id            Identifier of the input object
     * @param val           Value of the input object
     */
    public void add(InitStep2MasterCollectionInputId id, NumericTable val) {
        if (id == InitStep2MasterCollectionInputId.step2MeanDependentVariable || id == InitStep2MasterCollectionInputId.step2NumberOfRows ||
            id == InitStep2MasterCollectionInputId.step2BinBorders || id == InitStep2MasterCollectionInputId.step2BinSizes) {
            cAddNumericTable(this.cObject, id.getValue(), val.getCObject());
        } else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns an input object for the DBSCAN algorithm in the ninth step of the distributed processing mode
     * @param id      Identifier of the input object
     * @return        %Input object that corresponds to the given identifier
     */
    public DataCollection get(InitStep2MasterCollectionInputId id) {
        if (id == InitStep2MasterCollectionInputId.step2MeanDependentVariable || id == InitStep2MasterCollectionInputId.step2NumberOfRows ||
            id == InitStep2MasterCollectionInputId.step2BinBorders || id == InitStep2MasterCollectionInputId.step2BinSizes) {
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
