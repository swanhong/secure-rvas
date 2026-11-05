package crypto

import (
	"bufio"
	"encoding/binary"
	"io"
	"os"
)

const cipherSizeBytes = 8

func MarshalCM(cm CipherMatrix) ([]byte, []byte) {
	cmBytes, ctSizes, err := cm.MarshalBinary()
	if err != nil {
		panic(err)
	}

	r, c := len(cm), len(cm[0])
	sizesbuf := make([]byte, cipherSizeBytes*r*c)
	offset := 0
	for i := range ctSizes {
		for j := range ctSizes[i] {
			binary.LittleEndian.PutUint64(sizesbuf[offset:offset+cipherSizeBytes], uint64(ctSizes[i][j]))
			offset += cipherSizeBytes
		}
	}

	return sizesbuf, cmBytes
}

func UnmarshalCM(cryptoParams *CryptoParams, r, c int, sbytes, ctbytes []byte) CipherMatrix {
	offset := 0
	sizes := make([][]int, r)
	for i := range sizes {
		sizes[i] = make([]int, c)
		for j := range sizes[i] {
			sizes[i][j] = int(binary.LittleEndian.Uint64(sbytes[offset:]))
			offset += cipherSizeBytes
		}
	}

	var cm CipherMatrix
	if err := (&cm).UnmarshalBinary(cryptoParams, ctbytes, sizes); err != nil {
		panic(err)
	}

	return cm
}

func SaveCipherMatrixToFile(cm CipherMatrix, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	writer := bufio.NewWriter(file)
	sbytes, cmbytes := MarshalCM(cm)

	write := func(data []byte) {
		if _, err := writer.Write(data); err != nil {
			panic(err)
		}
	}

	header := make([]byte, 16)
	binary.LittleEndian.PutUint32(header, uint32(len(cm)))
	binary.LittleEndian.PutUint32(header[4:], uint32(len(cm[0])))
	binary.LittleEndian.PutUint64(header[8:], uint64(len(sbytes)))
	write(header)
	write(sbytes)
	binary.LittleEndian.PutUint64(header[:8], uint64(len(cmbytes)))
	write(header[:8])
	write(cmbytes)

	if err := writer.Flush(); err != nil {
		panic(err)
	}
}

func LoadCipherMatrixFromFile(cps *CryptoParams, filename string) CipherMatrix {
	file, err := os.Open(filename)
	if err != nil {
		panic(err)
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	read := func(data []byte) {
		if _, err := io.ReadFull(reader, data); err != nil {
			panic(err)
		}
	}

	header := make([]byte, 16)
	read(header)
	nrows := int(binary.LittleEndian.Uint32(header))
	numCtxPerRow := int(binary.LittleEndian.Uint32(header[4:]))
	sbyteSize := binary.LittleEndian.Uint64(header[8:])
	sdata := make([]byte, sbyteSize)
	read(sdata)

	read(header[:8])
	cbyteSize := binary.LittleEndian.Uint64(header)
	cdata := make([]byte, cbyteSize)
	read(cdata)

	return UnmarshalCM(cps, nrows, numCtxPerRow, sdata, cdata)
}
