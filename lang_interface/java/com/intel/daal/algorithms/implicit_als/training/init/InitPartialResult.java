/* file: InitPartialResult.java */
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
 * @ingroup implicit_als_init_distributed
 * @{
 */
package com.intel.daal.algorithms.implicit_als.training.init;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.implicit_als.PartialModel;
import com.intel.daal.services.DaalContext;

import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__IMPLICIT_ALS__TRAINING__INIT__INITPARTIALRESULT"></a>
 * @brief Provides methods to access partial results of computing the initial model for the
 * implicit ALS training algorithm
 */
public final class InitPartialResult extends InitPartialResultBase {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the partial result of the implicit ALS initialization algorithm in the distributed processing mode
     * @param context Context to manage the partial result
     */
    public InitPartialResult(DaalContext context) {
        super(context);
        this.cObject = cNewPartialResult();
    }

    public InitPartialResult(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Returns a partial result of computing the initial model for the implicit ALS training algorithm
     * @param  id   Identifier of the partial result
     * @return      Partial result that corresponds to the given identifier
     */
    public PartialModel get(InitPartialResultId id) {
        if (id != InitPartialResultId.partialModel) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new PartialModel(getContext(), cGetPartialResultModel(cObject, id.getValue()));
    }

    /**
     * Sets a partial result of computing the initial model for the implicit ALS training algorithm
     * @param id    Identifier of the partial result
     * @param value Partial result that corresponds to the given identifier
     */
    public void set(InitPartialResultId id, PartialModel value) {
        int idValue = id.getValue();
        if (id != InitPartialResultId.partialModel) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultModel(cObject, idValue, value.getCObject());
    }

    /**
     * Gets a KeyValueDataCollection partial result of the implicit ALS initialization algorithm
     * @param  id   Identifier of the partial result
     * @return      Partial result that corresponds to the given identifier
     */
    public KeyValueDataCollection get(InitPartialResultCollectionId id) {
        if (id != InitPartialResultCollectionId.outputOfStep1ForStep2) {
            throw new IllegalArgumentException("id unsupported");
        }
        return new KeyValueDataCollection(getContext(), cGetPartialResultCollection(cObject, id.getValue()));
    }

    /**
     * Gets a numeric table object from a partial result of the implicit ALS initialization algorithm
     * @param  id   Identifier of the partial result
     * @param key   Key to use to retrieve a numeric table
     * @return      Partial result that corresponds to the given identifier
     */
    public NumericTable get(InitPartialResultCollectionId id, long key) {
        if (id != InitPartialResultCollectionId.outputOfStep1ForStep2) {
            throw new IllegalArgumentException("id unsupported");
        }
        return (NumericTable)Factory.instance().createObject(getContext(), cGetPartialResultTable(cObject, id.getValue(), key));
    }

    /**
     * Sets the KeyValueDataCollection partial result of the implicit ALS initialization algorithm
     * @param id    Identifier of the partial result
     * @param value Partial result that corresponds to the given identifier
     */
    public void set(InitPartialResultCollectionId id, KeyValueDataCollection value) {
        int idValue = id.getValue();
        if (id != InitPartialResultCollectionId.outputOfStep1ForStep2) {
            throw new IllegalArgumentException("id unsupported");
        }
        cSetPartialResultCollection(cObject, idValue, value.getCObject());
    }

    private native long cNewPartialResult();

    private native long cGetPartialResultModel(long cResult, int id);
    private native void cSetPartialResultModel(long cResult, int id, long cModel);

    private native long cGetPartialResultCollection(long cResult, int id);
    private native void cSetPartialResultCollection(long cResult, int id, long cModel);

    private native long cGetPartialResultTable(long cResult, int id, long key);
}
/** @} */
