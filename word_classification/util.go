package word_classification

import (
	"bufio"
)

func FilterDictionary(dict Dictionary, max uint64) {
	for word, m := range dict {
		var count uint64 = 0

		for _, freq := range m {
			count += freq
		}

		if count > max {
			// XXX - delete
			dict[word] = make(map[string]uint64)
		}
	}
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
