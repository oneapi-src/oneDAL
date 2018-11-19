/* file: lzocompression.h */
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

/*
//++
//  Implementation of the LZO1X_11 compression and decompression interface.
//--
*/

#ifndef __LZOCOMPRESSION_H__
#define __LZOCOMPRESSION_H__
#include "data_management/compression/compression.h"

namespace daal
{
namespace data_management
{

namespace interface1
{
/**
 * @ingroup data_compression
 * @{
 */
/**
 * <a name="DAAL-CLASS-LZOCOMPRESSIONPARAMETER"></a>
 *
 * \brief Parameter for LZO compression and decompression.
 * LZO compressed block header consists of four sections: 1) optional, 2) uncompressed data size (4 bytes),
 * 3) compressed data size (4 bytes), 4) optional.
 *
 * \snippet compression/lzocompression.h LzoCompressionParameter source code
 *
 */
/* [LzoCompressionParameter source code] */
class DAAL_EXPORT LzoCompressionParameter : public data_management::CompressionParameter
{
public:
    /**
     * %LzoCompressionParameter constructor
     * \param _preHeadBytes  Size in bytes of section 1 of the LZO compressed block header
     * \param _postHeadBytes Size in bytes of section 4 of the LZO compressed block header
     */
    LzoCompressionParameter( size_t _preHeadBytes = 0, size_t _postHeadBytes = 0 ) :
        data_management::CompressionParameter(defaultLevel), preHeadBytes(_preHeadBytes), postHeadBytes(_postHeadBytes)
    {}
    ~LzoCompressionParameter() {}

    size_t preHeadBytes;   /*!< Size in bytes of section 1 of the LZO compressed block header */
    size_t postHeadBytes;  /*!< Size in bytes of section 4 of the LZO compressed block header */
};
/* [LzoCompressionParameter source code] */

/**
 * <a name="DAAL-CLASS-COMPRESSOR_LZO"></a>
 *
 * \brief Implementation of the Compressor class for the LZO compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref LzoCompressionParameter class
 */
template<> class DAAL_EXPORT Compressor<lzo> : public data_management::CompressorImpl
{
public:
    /**
     * \brief Compressor<lzo> constructor
     */
    Compressor();
    ~Compressor();
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Pointer to the data block to compress. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to compress in inBlock
     * \param[in] offset   Offset in bytes, the starting position for compression in inBlock
     */
    void setInputDataBlock( byte *inBlock, size_t size, size_t offset );
    /**
     * Associates an input data block with a compressor
     * \param[in] inBlock Reference to the data block to compress
     */
    void setInputDataBlock( DataBlock &inBlock )
    {
        setInputDataBlock( inBlock.getPtr(), inBlock.getSize(), 0 );
    }

    /**
     * Performs LZO compression of a data block
     * \param[out] outBlock Pointer to the data block where compression results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for compression in outBlock
     */
    void run( byte *outBlock, size_t size, size_t offset );
    /**
     * Performs LZO compression of a data block
     * \param[out] outBlock Reference to the data block where compression results are stored
     */
    void run( DataBlock &outBlock )
    {
        run( outBlock.getPtr(), outBlock.getSize(), 0 );
    }

    LzoCompressionParameter parameter; /*!< LZO compression parameters structure */

protected:
    void initialize();

private:
    void *_next_in;
    size_t _avail_in;
    void *_next_out;
    size_t _avail_out;
    void *_p_lzo_state;
    size_t _preHeadBytes;
    size_t _postHeadBytes;

    void finalizeCompression();
};

/**
 * <a name="DAAL-CLASS-DECOMPRESSOR_LZO"></a>
 *
 * \brief Specialization of Decompressor class for LZO compression method
 * <!-- \n<a href="DAAL-REF-COMPRESSION">Data compression usage model</a> -->
 *
 * \par References
 *      - \ref services::ErrorCompressionNullInputStream "Data compression error codes"
 *      - \ref LzoCompressionParameter class
 */
template<> class DAAL_EXPORT Decompressor<lzo> : public data_management::DecompressorImpl
{
public:
    /**
     * \brief Decompressor<lzo> constructor
     */
    Decompressor();
    ~Decompressor();
    /**
     * Associates an input data stream with a decompressor
     * \param[in] inBlock Pointer to the data block to decompress. Must be at least size+offset bytes
     * \param[in] size     Number of bytes to decompress in inBlock
     * \param[in] offset   Offset in bytes, the starting position for decompression in inBlock
     */
    void setInputDataBlock( byte *inBlock, size_t size, size_t offset );

    /**
     * Associates an input data stream with a decompressor
     * \param[in] inBlock Reference to the data block to decompress
     */
    void setInputDataBlock( DataBlock &inBlock )
    {
        return setInputDataBlock( inBlock.getPtr(), inBlock.getSize(), 0 );
    }

    /**
     * Performs LZO decompression of a data block
     * \param[out] outBlock Pointer to the data block where decompression results are stored. Must be at least size+offset bytes
     * \param[in] size       Number of bytes available in outBlock
     * \param[in] offset     Offset in bytes, the starting position for decompression in outBlock
     */
    void run( byte *outBlock, size_t size, size_t offset );

    /**
     * Performs LZO decompression of a data block
     * \param[out] outBlock Reference to the data block where decompression results are stored
     */
    void run( DataBlock &outBlock )
    {
        run( outBlock.getPtr(), outBlock.getSize(), 0 );
    }

    LzoCompressionParameter parameter; /*!< LZO compression parameters structure */

protected:
    void initialize();

private:
    void *_next_in;
    size_t _avail_in;
    void *_next_out;
    size_t _avail_out;
    void *_p_lzo_state;
    size_t _preHeadBytes;
    size_t _postHeadBytes;

    void *_internalBuff;
    size_t _internalBuffOff;
    size_t _internalBuffLen;

    void finalizeCompression();
};
/** @} */
} // namespace interface1
using interface1::LzoCompressionParameter;
using interface1::Compressor;
using interface1::Decompressor;

} //namespace data_management
} //namespace daal
#endif //__LZOCOMPRESSION_H
