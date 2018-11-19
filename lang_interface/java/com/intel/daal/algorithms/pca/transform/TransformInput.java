/* file: TransformInput.java */
/*******************************************************************************
* Copyright 2014-2018 Intel Corporation.
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
 * @ingroup pca_transform
 * @{
 */
/**
 * @brief Contains classes for computing the PCA transformation
 */
package com.intel.daal.algorithms.pca.transform;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.data_management.data.Factory;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.KeyValueDataCollection;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__PCA__TRANSFORM__INPUT"></a>
 * @brief %Input objects for the PCA transformation algorithm
 */
public final class TransformInput extends com.intel.daal.algorithms.Input {
    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    public TransformInput(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Sets the input object of the PCA transformation algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(TransformInputId id, NumericTable val) {
        if (id == TransformInputId.data || id == TransformInputId.eigenvectors) {
            cSetInputTable(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the PCA transformation algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public NumericTable get(TransformInputId id) {
        if (id == TransformInputId.data || id == TransformInputId.eigenvectors) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTable(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the input object of the PCA transformation algorithm
     * @param id    Identifier of the input object
     * @param val   Value of the input object
     */
    public void set(TransformDataInputId id, KeyValueDataCollection val) {
        if (id == TransformDataInputId.dataForTransform) {
            cSetInputTransformData(cObject, id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the PCA transformation algorithm
     * @param id Identifier of the input object
     * @return   Input object that corresponds to the given identifier
     */
    public KeyValueDataCollection get(TransformDataInputId id) {
        if (id == TransformDataInputId.dataForTransform) {
            return (KeyValueDataCollection)Factory.instance().createObject(getContext(), cGetInputTransformData(cObject, id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Sets the input object of the PCA transformation algorithm
     * @param wid   Identifier of the transform input object
     * @param id    Identifier of the component
     * @param val   Value of the input object
     */
    public void set(TransformDataInputId wid, TransformComponentId id, NumericTable val) {
        if (wid == TransformDataInputId.dataForTransform &&
           (id == TransformComponentId.eigenvalue || id == TransformComponentId.mean || id == TransformComponentId.variance)) {
                cSetInputTransformComponent(cObject, wid.getValue(), id.getValue(), val.getCObject());
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }

    /**
     * Returns the input object of the PCA transformation algorithm
     * @param wid   Identifier of the transform input object
     * @param id    Identifier of the component
     * @return      Input object that corresponds to the given identifier
     */
    public NumericTable get(TransformDataInputId wid, TransformComponentId id) {
        if (wid == TransformDataInputId.dataForTransform &&
           (id == TransformComponentId.eigenvalue || id == TransformComponentId.mean || id == TransformComponentId.variance)) {
            return (NumericTable)Factory.instance().createObject(getContext(), cGetInputTransformComponent(cObject, wid.getValue(), id.getValue()));
        }
        else {
            throw new IllegalArgumentException("id unsupported");
        }
    }


    private native void cSetInputTable(long cObject, int id, long ntAddr);
    private native long cGetInputTable(long cObject, int id);
    private native void cSetInputTransformData(long cObject, int id, long ntAddr);
    private native long cGetInputTransformData(long cObject, int id);
    private native void cSetInputTransformComponent(long cObject, int wid, int id, long ntAddr);
    private native long cGetInputTransformComponent(long cObject, int wid, int id);

}
/** @} */
