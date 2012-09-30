package word_classification

import (
	"encoding/json"
	"fmt"
	"github.com/danieldk/golinear"
	"io"
	"io/ioutil"
)

type StringFeature struct {
	Feature string
	Value   float64
}

type ModelMetadata struct {
	FeatureMapping map[string]int
	ClassMapping   map[string]int
	Normalizer     float64
}

func LoadMetadata(reader io.Reader) (*ModelMetadata, error) {
	bMetaData, err := ioutil.ReadAll(reader)
	if err != nil {
		return nil, err
	}

	var metadata ModelMetadata
	err = json.Unmarshal(bMetaData, &metadata)
	if err != nil {
		return nil, err
	}

	return &metadata, nil
}

func reverse(s string) string {
	runes := []rune(s)
	for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
		runes[i], runes[j] = runes[j], runes[i]
	}
	return string(runes)
}

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

var DefaultTemplates = []FeatureTemplate{Prefixes(4), Suffixes(4)}

func ApplyTemplates(templates []FeatureTemplate, word string) []StringFeature {
	features := make([]StringFeature, 0)
	for _, template := range templates {
		features = append(features, template(word)...)
	}

	return features
}

func StringFeatureToFeature(features []StringFeature, mapping map[string]int, norm float64) []golinear.FeatureValue {
	numberedFeatures := make([]golinear.FeatureValue, len(features))

	for idx, fv := range features {
		id, found := mapping[fv.Feature]
		if !found {
			mapping[fv.Feature] = len(mapping) + 1
		}

		id, _ = mapping[fv.Feature]
		numberedFeatures[idx].Index = id
		numberedFeatures[idx].Value = fv.Value / norm
	}

	return numberedFeatures
}

func ExtractFeatures(dict Dictionary) (*golinear.Problem, ModelMetadata) {
	problem := golinear.NewProblem()
	//featureMap := make(map[string]int)
	featureMapping := make(map[string]int)
	tagMapping := make(map[string]int)

	var norm uint64 = 0
	for _, tags := range dict {
		for _, count := range tags {
			if count > norm {
				norm = count
			}
		}
	}

	for word, tags := range dict {
		featureVec := StringFeatureToFeature(ApplyTemplates(DefaultTemplates, word),
			featureMapping, float64(norm))

		for tag, count := range tags {
			id, found := tagMapping[tag]
			if !found {
				tagMapping[tag] = len(tagMapping)
			}

			id, _ = tagMapping[tag]

			for i := 0; i < int(count); i++ {
				var inst golinear.TrainingInstance

				inst.Features = featureVec
				inst.Label = float64(id)

				if err := problem.Add(inst); err != nil {
					panic(err)
				}
			}
		}
	}

	return problem, ModelMetadata{featureMapping, tagMapping, float64(norm)}
}
