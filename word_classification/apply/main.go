package main

import (
	"bufio"
	"fmt"
	"github.com/danieldk/golinear"
	"github.com/danieldk/golinear-examples/word_classification"
	"log"
	"os"
)

func loadMetadata(modelBasename string) (*word_classification.ModelMetadata, error) {
	f, err := os.Open(fmt.Sprintf("%s.metadata", modelBasename))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	reader := bufio.NewReader(f)

	return word_classification.LoadMetadata(reader)
}

func reverseMapping(mapping map[string]int) map[int]string {
	reverse := make(map[int]string)

	for k, v := range mapping {
		reverse[v] = k
	}

	return reverse
}

func main() {
	if len(os.Args) != 3 {
		fmt.Printf("%s model word\n", os.Args[0])
		os.Exit(1)
	}

	modelBasename := os.Args[1]
	model, err := golinear.LoadModel(fmt.Sprintf("%s.model", modelBasename))
	if err != nil {
		log.Fatal(err)
	}

	metadata, err := loadMetadata(modelBasename)
	if err != nil {
		log.Fatal(err)
	}

	word := os.Args[2]
	sfs := word_classification.ApplyTemplates(word_classification.DefaultTemplates, word)
	//	fmt.Printf("%#v", sfs)
	fs := word_classification.StringFeatureToFeature(sfs, metadata.FeatureMapping, metadata.Normalizer)

	class := model.Predict(fs)

	indexToClass := reverseMapping(metadata.ClassMapping)

	fmt.Printf("Predicted class: %s\n", indexToClass[int(class)])
}
