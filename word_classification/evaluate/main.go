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

func readDictionary(filename string) word_classification.Dictionary {
	f, err := os.Open(filename)
	if err != nil {
		fmt.Printf("Could not open file: %s\n", filename)
		os.Exit(1)
	}

	r := bufio.NewReader(f)
	return word_classification.ReadDictionary(r)
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

	testDict := readDictionary(os.Args[2])
	word_classification.FilterDictionary(testDict, 6)

	indexToClass := reverseMapping(metadata.ClassMapping)

	total, correct, baseline := 0, 0, 0

	errorsPerClass := make(map[string]int)
	totalPerClass := make(map[string]int)

	for word, tagFreq := range testDict {
		sfs := append(word_classification.Prefixes(word, 3), word_classification.Suffixes(word, 3)...)
		fs := word_classification.StringFeatureToFeature(sfs, metadata.FeatureMapping, metadata.Normalizer)

		for tag, freq := range tagFreq {
			var i uint64
			for i = 0; i < freq; i++ {
				class := model.Predict(fs)

				predictedTag := indexToClass[int(class)]

				// Update counts
				if predictedTag == tag {
					correct++
				} else {
					errorsPerClass[predictedTag]++
				}

				if tag == "NN" {
					baseline++
				}

				total++
				totalPerClass[predictedTag]++
			}
		}
	}

	errors := total - correct
	for class, classErrors := range errorsPerClass {
		fmt.Printf("%s:\t\t%.2f%%\t%.2f%%\n", class,
			float64(classErrors)/float64(errors)*100,
			float64(classErrors)/float64(totalPerClass[class])*100)
	}

	//fmt.Printf("Correct classifications: %d, total: %d\n", correct, total)
	fmt.Printf("\nAccuracy: %.2f%%\n", float64(correct)/float64(total)*100)
	fmt.Printf("Baseline: %.2f%%\n", float64(baseline)/float64(total)*100)
}
