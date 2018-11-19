/* file: AveragePooling3dForwardBatch.java */
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
 * @defgroup average_pooling3d_forward_batch Batch
 * @ingroup average_pooling3d_forward
 * @{
 */
package com.intel.daal.algorithms.neural_networks.layers.average_pooling3d;

import com.intel.daal.utils.*;
import com.intel.daal.algorithms.Precision;
import com.intel.daal.algorithms.ComputeMode;
import com.intel.daal.algorithms.AnalysisBatch;
import com.intel.daal.services.DaalContext;
import com.intel.daal.algorithms.neural_networks.layers.pooling3d.Pooling3dIndices;

/**
 * <a name="DAAL-CLASS-ALGORITHMS__NEURAL_NETWORKS__LAYERS__AVERAGE_POOLING3D__AVERAGEPOOLING3DFORWARDBATCH"></a>
 * \brief Class that computes the results of the forward three-dimensional average pooling layer in the batch processing mode
 * <!-- \n<a href="DAAL-REF-AVERAGEPOOLING3DFORWARD-ALGORITHM">Forward three-dimensional average pooling layer description and usage models</a> -->
 *
 * \par References
 *      - @ref AveragePooling3dLayerDataId class
 */
public class AveragePooling3dForwardBatch extends com.intel.daal.algorithms.neural_networks.layers.ForwardLayer {
    public  AveragePooling3dForwardInput input;     /*!< %Input data */
    public  AveragePooling3dMethod       method;    /*!< Computation method for the layer */
    public  AveragePooling3dParameter    parameter; /*!< AveragePooling3dParameters of the layer */
    private Precision                    prec;      /*!< Data type to use in intermediate computations for the layer */

    /** @private */
    static {
        LibUtils.loadLibrary();
    }

    /**
     * Constructs the forward three-dimensional average pooling layer by copying input objects of another forward
     * three-dimensional average pooling layer
     * @param context    Context to manage the forward three-dimensional average pooling layer
     * @param other      A forward three-dimensional average pooling layer to be used as the source to initialize
     *                   the input objects of the forward three-dimensional average pooling layer
     */
    public AveragePooling3dForwardBatch(DaalContext context, AveragePooling3dForwardBatch other) {
        super(context);
        this.method = other.method;
        prec = other.prec;

        this.cObject = cClone(other.cObject, prec.getValue(), method.getValue());
        input = new AveragePooling3dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Constructs the forward three-dimensional average pooling layer
     * @param context    Context to manage the layer
     * @param cls        Data type to use in intermediate computations for the layer, Double.class or Float.class
     * @param method     The layer computation method, @ref AveragePooling3dMethod
     * @param nDim       Number of dimensions in input data
     */
    public AveragePooling3dForwardBatch(DaalContext context, Class<? extends Number> cls, AveragePooling3dMethod method, long nDim) {
        super(context);

        this.method = method;

        if (method != AveragePooling3dMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cInit(prec.getValue(), method.getValue(), nDim);
        input = new AveragePooling3dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
    }

    AveragePooling3dForwardBatch(DaalContext context, Class<? extends Number> cls, AveragePooling3dMethod method, long cObject, long nDim) {
        super(context);

        this.method = method;

        if (method != AveragePooling3dMethod.defaultDense) {
            throw new IllegalArgumentException("method unsupported");
        }
        if (cls != Double.class && cls != Float.class) {
            throw new IllegalArgumentException("type unsupported");
        }

        if (cls == Double.class) {
            prec = Precision.doublePrecision;
        }
        else {
            prec = Precision.singlePrecision;
        }

        this.cObject = cObject;
        input = new AveragePooling3dForwardInput(context, cGetInput(cObject, prec.getValue(), method.getValue()));
        parameter = new AveragePooling3dParameter(context, cInitParameter(cObject, prec.getValue(), method.getValue()));
        Pooling3dIndices sd = new Pooling3dIndices(nDim - 3, nDim - 2, nDim - 1);
        parameter.setIndices(sd);
    }

    /**
     * Computes the result of the forward three-dimensional average pooling layer
     * @return  Forward three-dimensional average pooling layer result
     */
    @Override
    public AveragePooling3dForwardResult compute() {
        super.compute();
        AveragePooling3dForwardResult result = new AveragePooling3dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
        return result;
    }

    /**
     * Registers user-allocated memory to store the result of the forward three-dimensional average pooling layer
     * @param result    Structure to store the result of the forward three-dimensional average pooling layer
     */
    public void setResult(AveragePooling3dForwardResult result) {
        cSetResult(cObject, prec.getValue(), method.getValue(), result.getCObject());
    }

    /**
     * Returns the structure that contains result of the forward layer
     * @return Structure that contains result of the forward layer
     */
    @Override
    public AveragePooling3dForwardResult getLayerResult() {
        return new AveragePooling3dForwardResult(getContext(), cGetResult(cObject, prec.getValue(), method.getValue()));
    }

    /**
     * Returns the structure that contains input object of the forward layer
     * @return Structure that contains input object of the forward layer
     */
    @Override
    public AveragePooling3dForwardInput getLayerInput() {
        return input;
    }

    /**
     * Returns the structure that contains parameters of the forward layer
     * @return Structure that contains parameters of the forward layer
     */
    @Override
    public AveragePooling3dParameter getLayerParameter() {
        return parameter;
    }

    /**
     * Returns the newly allocated forward three-dimensional average pooling layer
     * with a copy of input objects of this forward three-dimensional average pooling layer
     * @param context    Context to manage the layer
     *
     * @return The newly allocated forward three-dimensional average pooling layer
     */
    @Override
    public AveragePooling3dForwardBatch clone(DaalContext context) {
        return new AveragePooling3dForwardBatch(context, this);
    }

    private native long cInit(int prec, int method, long nDim);
    private native long cInitParameter(long cAlgorithm, int prec, int method);
    private native long cGetInput(long cAlgorithm, int prec, int method);
    private native long cGetResult(long cAlgorithm, int prec, int method);
    private native void cSetResult(long cAlgorithm, int prec, int method, long cObject);
    private native long cClone(long algAddr, int prec, int method);
}
/** @} */
