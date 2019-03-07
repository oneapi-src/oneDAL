/* file: PreluParameter.java */
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
 * @ingroup prelu
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.prelu;

import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__PRELU__PRELUPARAMETER"></a>
 * \brief Class that specifies parameters of the prelu layer
 */
public class PreluParameter extends com.intel.daal.algorithms.neural_networks.layers.Parameter {

    /**
     * Constructs the parameter of the prelu layer
     * @param context   Context to manage the parameter of the prelu layer
     */
    public PreluParameter(DaalContext context) {
        super(context);
        cObject = cInit();
    }

    public PreluParameter(DaalContext context, long cParameter) {
        super(context, cParameter);
    }

    /**
     *  Gets the index of the dimension for which the weights are applied
     */
    public long getDataDimension() {
        return cGetDataDimension(cObject);
    }

    /**
     *  Sets the index of the dimension for which the weights are applied
     *  @param dataDimension   Starting data dimension index to apply weight
     */
    public void setDataDimension(long dataDimension) {
        cSetDataDimension(cObject, dataDimension);
    }

    /**
    *  Gets the number of weights dimension
    */
    public long getWeightsDimension() {
        return cgetWeightsDimension(cObject);
    }

    /**
     *  Sets the number of weights dimension
     *  @param weightsDimension   The number of weights dimension
     */
    public void setWeightsDimension(long weightsDimension) {
        csetWeightsDimension(cObject, weightsDimension);
    }

    private native long    cInit();
    private native long cGetDataDimension(long cParameter);
    private native void cSetDataDimension(long cParameter, long dataDimension);
    private native long cgetWeightsDimension(long cParameter);
    private native void csetWeightsDimension(long cParameter, long weightsDimension);
}
/** @} */
