# Repository for ICSME'2022 Paper "PEELER: Learning to Effectively Predict Flakiness without Running Tests"

## Summary
*Peeler* is a fully static flaky tests detector that relies on data dependency in a Test Dependency Graph. *Peeler* captures test flakiness by modeling how the values of the variables involved in assertion statements are transferred as reflected in the embedded contextual paths. 
The evaluation on 17,532 tests from 21 projects (of which 689 are flaky) shows that *Peeler* outperforms prior works by about 20% in terms of Precision and F-score. During a live study, *Peelr* has already helped developers identify 12 flaky test cases in real-world project test suites.

## Environment
- OS: Ubuntu 18.04
- Python version: >= 3.6.13
- Java JDK version: >= 11.0.2
- CUDA version: 11.1

## Experiments
### Benchmark
Before extracting contextual paths from flaky/non-flaky tests, a csv file contain the information of the flaky/non-flaky tests is required. Generate `FlakeFlagger_benchmark.csv` from the FlakeFlagger dataset through the following command:
```shell
python dataset/benchmark/transform_FlakeFlagger_dataset.py
```
Notice that this process will clone all of the java projects related with the dataset, the projects directory is set in `transform_FlakeFlagger_dataset.py`.
Also, you can build your own dataset in the form of `FlakeFlagger_benchmark.csv`.

### Corpus
Once we got the benchmark, the features of flaky/non-flaky tests is able to be extracted by the *pathextractor*, which generates a TDG for each flaky/non-flaky method and extracts a set of contextual paths from it. Command:
```shell
java -jar pathextractor/target/pathextractor-1.0-SNAPSHOT-jar-with-dependencies.jar --csvFile <benchmark csv file>  --projectsDir <projects dir> --output_dir <output dir>
```
A file named `corpus.txt` will be generated under \<output dir\>, which contains features of all flaky/non-flaky tests of the dataset.

### Dataset
Parse `corpus.txt` to final dataset through command:
```shell
python corpus/parse_corpus.py --file <corpus.txt file> --output_dir <output dir>
```
Three files named `functions.txt`(edge information in paths), `samples.txt`(paths of each flaky/non-flaky tests) and `tokens.txt`(token dictionary) respectively will be generated under \<output dir\>.

### Pretrained Embeddings
We utilize pretrained [**code2vec**](https://github.com/tech-srl/code2vec) to generate embeddings for tokens and functions. The commands are:
```shell
python3 embeddings/method2embeddings.py --input_file <functions.txt> --code2vec_dir <code2vec project dir> --output_dir <output dir>
python3 embeddings/tokens2embeddings.py --file <tokens.txt> --vecs <pretrained token embeddings> --output_dir <output dir>

```
The pretrained token embeddings we used can be downloaded [here](https://s3.amazonaws.com/code2vec/model/token_vecs.tar.gz), the pretrained model can be downloaded [here](https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz).
`methods_embeddings.txt` and `tokens_embeddings.txt` will be generated under \<output dir\>.

### Train Peeler
Run the following command to train a Peeler model from scratch, detailed parameters can be seen in `flaky_detect.py`:
```shell
python flaky_detect.py
```
`test_result.csv` and the trained `.model` file will be generated under output directory.
Training logs will be found in `logs` directory.

### Evaluate Peeler
Evaluate pretrained Peeler model through this command:
```shell
python flaky_detect.py --predict_model <pretrained model> --predict_mode True
```

## Evaluation Results
The training logs and the test results for 10 fold validation are released in `logs` and `outputs` directories respectively. More evaluation results can be found [here](https://doi.org/10.5281/zenodo.5401937).
