/* file: LcnParameter.java */
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
 * @ingroup lcn_layers
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.lcn;

import com.intel.daal.services.DaalContext;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.data_management.data.Tensor;
import com.intel.daal.data_management.data.Factory;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__LCN__LCNPARAMETER"></a>
 * \brief Class that specifies parameters of the local contrast normalization layer
 */
public class LcnParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     * Constructs the parameter of the local contrast normalization layer
     * @param context Context to manage the parameter of the local contrast normalization layer
     */
    public LcnParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }
    /**
     * Constructs parameter from C++ parameter
     * @param context Context to manage the parameter
     * @param cObject Address of C++ parameter
     */
    public LcnParameter(DaalContext context, long cObject) {
        super(context, cObject);
    }

    /**
     * Gets the structure representing the indices of the two dimensions on which local contrast normalization is performed
     * @return The structure representing the indices of the two dimensions on which local contrast normalization is performed
     */
    public LcnIndices getIndices() {
        long[] dims = cGetIndices(cObject);
        return new LcnIndices(dims[0], dims[1]);
    }

    /**
     * Sets the structure representing the indices of the two dimensions on which local contrast normalization is performed
     * @param indices   The structure representing the indices of the two dimensions on which local contrast normalization is performed
     */
    public void setIndices(LcnIndices indices) {
        long[] dims = indices.getSize();
        cSetIndices(cObject, dims[0], dims[1]);
    }

    /**
     *  Gets the numeric table of size 1x1 that stores dimension f
     */
    public NumericTable getSumDimension() {
        return (NumericTable)Factory.instance().createObject(getContext(), cGetSumDimension(cObject));
    }

    /**
     *  Sets the numeric table of size 1x1 that stores dimension f
     *  @param sumDimension   Numeric table of size 1x1 that stores dimension f
     */
    public void setSumDimension(NumericTable sumDimension) {
        cSetSumDimension(cObject, sumDimension.getCObject());
    }

    /**
     *  Gets the tensor of the two-dimensional kernel
     */
    public Tensor getKernel() {
        return (Tensor)Factory.instance().createObject(getContext(), cGetKernel(cObject));
    }

    /**
     *  Sets the tensor of the two-dimensional kernel
     *  @param kernel   Tensor of the two-dimensional kernel
     */
    public void setKernel(Tensor kernel) {
        cSetKernel(cObject, kernel.getCObject());
    }

    private native long cInit();
    private native void cSetIndices(long cObject, long first, long second);
    private native long[] cGetIndices(long cObject);
    private native long cGetSumDimension(long cParameter);
    private native void cSetSumDimension(long cParameter, long sumDimension);
    private native long cGetKernel(long cParameter);
    private native void cSetKernel(long cParameter, long kernel);
}
/** @} */
