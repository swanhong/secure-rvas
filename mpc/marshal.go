package mpc

import (
	"encoding/binary"
	"fmt"

	mpc_core "github.com/hhcho/mpc-core"
	"github.com/hhcho/sfgwas/crypto"
)

const marshalSizeEntryBytes = 8
const marshalPolySizeBytes = 4

func MarshalRData(val interface{}) []byte {
	switch t := val.(type) {
	case mpc_core.RElem:
		buf := make([]byte, t.NumBytes())
		t.ToBytes(buf)
		return buf
	case mpc_core.RVec:
		buf, _ := t.MarshalBinary()
		return buf
	case mpc_core.RMat:
		buf, _ := t.MarshalBinary()
		return buf
	default:
		panic(fmt.Sprintf("MarshalRData: unsupported type %T", t))
	}
}

//Wrappers for marshaling functions defined in crypto

// MarshalCV returns bytes for ct sizes and bytes for each chiphertext: 8 bytes used //TODO: check if 4 bytes are enough
func MarshalCV(cv crypto.CipherVector) ([]byte, []byte) {
	cvBytes, ctSizes, err := cv.MarshalBinary()
	if err != nil {
		panic(err)
	}
	sizesbuf := make([]byte, marshalSizeEntryBytes*len(ctSizes))
	offset := 0
	for i := range ctSizes {
		binary.LittleEndian.PutUint64(sizesbuf[offset:offset+marshalSizeEntryBytes], uint64(ctSizes[i]))
		offset += marshalSizeEntryBytes
	}
	return sizesbuf, cvBytes
}

// UnmarshalCipherVector updates cv to have the correct ciphertexts (cv must have the corrext size)
func UnmarshalCV(cryptoParams *crypto.CryptoParams, ncts int, sbytes, ctbytes []byte) crypto.CipherVector {
	if len(sbytes) != marshalSizeEntryBytes*ncts {
		panic(fmt.Sprintf("UnmarshalCV: invalid size buffer length %d for %d ciphertexts", len(sbytes), ncts))
	}
	sizes := make([]int, ncts)
	offset := 0
	for i := range sizes {
		sizes[i] = int(binary.LittleEndian.Uint64(sbytes[offset : offset+marshalSizeEntryBytes]))
		offset += marshalSizeEntryBytes
	}

	cv := make(crypto.CipherVector, ncts)
	if err := (&cv).UnmarshalBinary(cryptoParams, ctbytes, sizes); err != nil {
		panic(err)
	}
	return cv
}
