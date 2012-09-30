package word_classification

import (
	"fmt"
)

type FeatureTemplate func(string) []StringFeature

func GenericPrefix(n int, format, word string) []StringFeature {
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

func Prefixes(n int) FeatureTemplate {
	return func(word string) []StringFeature {
		return GenericPrefix(n, "prefix(%s)", word)
	}
}

func Suffixes(n int) FeatureTemplate {
	return func(word string) []StringFeature {
		reversed := reverse(word)
		return GenericPrefix(n, "suffix(%s)", reversed)
	}
}
