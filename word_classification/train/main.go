package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"github.com/danieldk/golinear"
	"github.com/danieldk/golinear-examples/word_classification"
	"log"
	"os"
)

func reverseMapping(mapping map[string]int) map[int]string {
	reverse := make(map[int]string)

	for k, v := range mapping {
		reverse[v] = k
	}

	return reverse
}

func main() {
	flag.Parse()

	if len(flag.Args()) != 2 {
		fmt.Printf("Usage: %s lexicon modelname\n", os.Args[0])
		os.Exit(1)
	}

	f, err := os.Open(flag.Arg(0))
	if err != nil {
		fmt.Printf("Could not open file: %s\n", flag.Arg(0))
		os.Exit(1)
	}

	r := bufio.NewReader(f)
	dict := word_classification.ReadDictionary(r)

	word_classification.FilterDictionary(dict, 6)

	problem, metadata := word_classification.ExtractFeatures(dict)

	param := golinear.DefaultParameters()

	model, err := golinear.TrainModel(param, problem)
	if err != nil {
		panic(err)
	}

	modelName := flag.Arg(1)

	err = model.Save(fmt.Sprintf("%s.model", modelName))
	if err != nil {
		panic(err)
	}

	bMetadata, err := json.Marshal(metadata)
	if err != nil {
		panic(err)
	}

	metadataFile, err := os.OpenFile(fmt.Sprintf("%s.metadata", modelName),
		os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		log.Fatal(err)
	}
	defer metadataFile.Close()

	metadataFile.Write(bMetadata)

	//testPrefix := prefixes("Microsoft", 3)
	//features := stringFeatureToFeature(testPrefix, featureMapping, norm)

	//class := model.Predict(features)

	//numberTagMapping := reverseMapping(tagMapping)

	//fmt.Printf("Predicted class: %s\n", numberTagMapping[int(class)])
}
