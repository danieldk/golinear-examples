package word_classification

import (
	"bufio"
)

func AppendFeatureLists(l1, l2 []StringFeature) []StringFeature {
	l := len(l1) + len(l2)

	newSlice := make([]StringFeature, l)
	copy(newSlice, l1)
	copy(newSlice[len(l1):], l2)

	return newSlice
}

func ReadLn(r *bufio.Reader) (string, error) {
	var (
		isPrefix bool  = true
		err      error = nil
		line, ln []byte
	)
	for isPrefix && err == nil {
		line, isPrefix, err = r.ReadLine()
		ln = append(ln, line...)
	}
	return string(ln), err
}
