/* file: Factory.java */
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
 * @ingroup serialization
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.SerializationTag;
import com.intel.daal.services.DaalContext;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__FACTORY"></a>
 * @brief Class that provides factory functionality for objects derived from the SerializableBase class.
 */
public class Factory {

    protected Factory() {}

    /**
     * Static function that returns an instance of the Factory class
     * @return Instance of the Factory object
     */
    static public Factory instance() {
        return new Factory();
    }

    /**
     * Creates a new object of Java class from a native object
     * @param  context Context for managing the memory in the native part of the partial result object
     * @param  cObject Pointer to the native object
     * @return Java object created from a native object
     */
    public SerializableBase createObject(DaalContext context, long cObject) {
        int objectId = cGetSerializationTag(cObject);
        if (objectId == SerializationTag.SERIALIZATION_SVM_MODEL_ID.getValue()) {
            return new com.intel.daal.algorithms.svm.Model(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_IMPLICIT_ALS_PARTIALMODEL_ID.getValue()) {
            return new com.intel.daal.algorithms.implicit_als.PartialModel(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_PARTIAL_RESULT_ID.getValue()) {
            return new com.intel.daal.algorithms.implicit_als.prediction.ratings.RatingsPartialResult(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_IMPLICIT_ALS_PREDICTION_RATINGS_RESULT_ID.getValue()) {
            return new com.intel.daal.algorithms.implicit_als.prediction.ratings.RatingsResult(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_DATACOLLECTION_ID.getValue()) {
            return new com.intel.daal.data_management.data.DataCollection(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_KEYVALUEDATACOLLECTION_ID.getValue()) {
            return new com.intel.daal.data_management.data.KeyValueDataCollection(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_CSR_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.CSRNumericTable(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_HOMOGEN_FLOAT32_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_HOMOGEN_FLOAT64_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_HOMOGEN_INT32_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_HOMOGEN_INT32_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_HOMOGEN_INT64_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_HOMOGEN_INT64_U_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.HomogenNumericTable(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_AOS_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.HomogenNumericTable(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_SOA_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.HomogenNumericTable(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_MATRIX_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.Matrix(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT32_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDSYMMETRIC_FLOAT64_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDSYMMETRIC_INT32_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDSYMMETRIC_INT64_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT32_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDSYMMETRIC_FLOAT64_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDSYMMETRIC_INT32_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDSYMMETRIC_INT64_U_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.PackedSymmetricMatrix(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT32_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDTRIANGULAR_FLOAT64_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDTRIANGULAR_INT32_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_UPPERPACKEDTRIANGULAR_INT64_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT32_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDTRIANGULAR_FLOAT64_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDTRIANGULAR_INT32_U_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_S_NT_ID.getValue() ||
            objectId == SerializationTag.SERIALIZATION_LOWERPACKEDTRIANGULAR_INT64_U_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.PackedTriangularMatrix(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_MERGE_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.MergedNumericTable(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_ROWMERGE_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.RowMergedNumericTable(context, cObject);
        }
        if (objectId == SerializationTag.SERIALIZATION_JAVANIO_CSR_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.CSRNumericTable(context, (CSRNumericTableImpl)cGetJavaNumericTable(cObject));
        }
        if (objectId == SerializationTag.SERIALIZATION_JAVANIO_HOMOGEN_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.HomogenNumericTable(context, (HomogenNumericTableImpl)cGetJavaNumericTable(cObject));
        }
        if (objectId == SerializationTag.SERIALIZATION_JAVANIO_AOS_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.AOSNumericTable(context, (AOSNumericTableImpl)cGetJavaNumericTable(cObject));
        }
        if (objectId == SerializationTag.SERIALIZATION_JAVANIO_SOA_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.SOANumericTable(context, (SOANumericTableImpl)cGetJavaNumericTable(cObject));
        }
        if (objectId == SerializationTag.SERIALIZATION_JAVANIO_PACKEDSYMMETRIC_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.PackedSymmetricMatrix(context, (PackedSymmetricMatrixImpl)cGetJavaNumericTable(cObject));
        }
        if (objectId == SerializationTag.SERIALIZATION_JAVANIO_PACKEDTRIANGULAR_NT_ID.getValue()) {
            return new com.intel.daal.data_management.data.PackedTriangularMatrix(context, (PackedTriangularMatrixImpl)cGetJavaNumericTable(cObject));
        }
        if (cObject == 0) {
            return null;
        }
        return new com.intel.daal.data_management.data.SerializableBase(context, cObject);
    }

    private native int cGetSerializationTag(long cObject);

    private native Object cGetJavaNumericTable(long cObject);
}
/** @} */
