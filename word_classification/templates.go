package word_classification

import (
	"fmt"
	"unicode"
)

type FeatureTemplate func(string) []StringFeature

func genericPrefix(n int, format, word string) []StringFeature {
	if len(word) < n {
		n = len(word)
	}

	features := make([]StringFeature, n)

	for i := 0; i < n; i++ {
		features[i].Feature = fmt.Sprintf(format, word[:i+1])
		features[i].Value = 1.0
	}

	return features
}

func countCharacters(f func(rune) bool, word string) uint {
	count := uint(0)

	for _, r := range word {
		if f(r) {
			count++
		}
	}

	return count
}

func Digits(word string) []StringFeature {
	count := countCharacters(unicode.IsDigit, word)
	return []StringFeature{StringFeature{"digits", float64(count)}}
}

func Capitals(word string) []StringFeature {
	count := countCharacters(unicode.IsUpper, word)
	return []StringFeature{StringFeature{"capitals", float64(count)}}
}

func Punct(word string) []StringFeature {
	count := countCharacters(unicode.IsPunct, word)
	return []StringFeature{StringFeature{"punct", float64(count)}}
}

func Prefixes(n int) FeatureTemplate {
	return func(word string) []StringFeature {
		return genericPrefix(n, "prefix(%s)", word)
	}
}

func Suffixes(n int) FeatureTemplate {
	return func(word string) []StringFeature {
		reversed := reverse(word)
		return genericPrefix(n, "suffix(%s)", reversed)
	}
}
