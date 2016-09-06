/* file: SerializableBase.java */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation
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

package com.intel.daal.data_management.data;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.nio.ByteBuffer;

import com.intel.daal.services.ContextClient;
import com.intel.daal.services.DaalContext;

/**
*  <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__SERIALIZABLEBASE"></a>
*  @brief Class that provides methods for serialization and deserialization
*/
abstract public class SerializableBase extends ContextClient
        implements Serializable, com.intel.daal.services.Disposable {

    /** @private */
    static {
        System.loadLibrary("JavaAPI");
    }

    public SerializableBase(DaalContext context) {
        super(context);
        this.cObject = 0;
        this.serializedCObject = null;
    }

    /**
     * Checks if the native object is deserialized
     */
    public void checkCObject() {
        if (this.cObject == 0 && this.serializedCObject != null) {
            throwUnpacked();
        }
    }

    /**
     * Returns the address of the native object
     *
     * @return Address of the native object
     */
    public long getCObject() {
        checkCObject();
        return this.cObject;
    }

    /**
     * Serializes an object
     */
    public void pack() {
        synchronized(this) {
            DaalContext myContext = getContext();
            if (myContext != null) {
                changeContext(null);
            }
            onPack();
            if (this.cObject != 0) {
                serializeCObject();
            }
            dispose();
        }
    }

    /**
     * Deserializes an object
     *
     * @param context   Context to manage a deserialized object
     */
    public void unpack(DaalContext context) {
        synchronized(this) {
            cSetJavaVM();
            changeContext(context);
            if (this.cObject == 0) {
                cSetDaalContext(context);
                onUnpack(context);
                cClearDaalContext();
            }
            this.serializedCObject = null;
        }
    }

    /**
     * Releases the memory allocated for the native object
     */
    @Override
    public void dispose() {
        if (this.cObject != 0) {
            cDispose(this.cObject);
            this.cObject = 0;
        }
    }

    /* --------------------- */
    /* Private and Protected */
    /* --------------------- */

    /* Pointer to SharedPtr<> for the C object */
    protected transient long cObject;

    /* Serialized C object */
    protected byte[][] serializedCObject;

    protected boolean onSerializeCObject() {
        return true;
    }

    protected void serializeCObject() {
        if (onSerializeCObject() && this.cObject != 0) {
            serializedCObject = cSerializeCObject(this.cObject);
            cObject = 0;
        }
    }

    protected void onPack() {}

    protected void onUnpack(DaalContext context) {
        deserializeCObject();
    }

    protected void deserializeCObject() {
        if (this.serializedCObject != null) {
            if (this.cObject != 0) {
                this.dispose();
            }
            this.cObject = cDeserializeCObject(serializedCObject);
        }
    }

    private void readObject(ObjectInputStream aInputStream) throws ClassNotFoundException, IOException {
        aInputStream.defaultReadObject();
        this.cObject = 0;
    }

    private void writeObject(ObjectOutputStream aOutputStream) throws IOException {
        aOutputStream.defaultWriteObject();
    }

    private native byte[][] cSerializeCObject(long cDataCollection);

    private native long cDeserializeCObject(byte[][] byteArray);

    private native void cFreeByteBuffer(ByteBuffer buffer);

    private native void cDispose(long parAddr);

    private native void throwUnpacked();

    private native void cSetJavaVM();

    private native void cSetDaalContext(DaalContext context);

    private native void cClearDaalContext();
}
