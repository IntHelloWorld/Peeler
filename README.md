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
Before extracting contextual paths from flaky/non-flaky tests, a csv file contain the information of the flaky/non-flaky tests is required. Generate `FlakeFlagger_benchmark.csv` from the [FlakeFlagger](https://zenodo.org/record/4450723#.YqksxaFBx3s) dataset through the following command:
```shell
python dataset/benchmark/transform_FlakeFlagger_dataset.py
```
Notice that this process will clone all of the java projects related with the dataset, the projects directory is set in `transform_FlakeFlagger_dataset.py`.
Also, you can build your own dataset in the form of `FlakeFlagger_benchmark.csv`.

### Corpus
Once we got the benchmark, the features of flaky/non-flaky tests is able to be extracted by the *pathextractor*, which generates a TDG for each flaky/non-flaky method and extracts a set of contextual paths from it. Command:
```shell
java -jar pathextractor/target/pathextractor-1.0-SNAPSHOT-jar-with-dependencies.jar --csvFile --projectsDir --output_dir
```
- `--csvFile`: the benchmark csv file contains the information of all the flaky/non-flaky tests. (e.g., `benchmark/FlakeFlagger_benchmark.csv`)
- `--projectsDir`: the directory to the cloned projects in step **Benchmark**. (e.g., `~/projects`)
- `--output_dir`: the directory where a file named `corpus.txt` will be generated, which contains features of all flaky/non-flaky tests of the dataset. (e.g., `corpus`)

### Dataset
Parse `corpus.txt` to final dataset through command:
```shell
python corpus/parse_corpus.py --file --output_dir
```
- `--file`: path to the `corpus.txt` generated in step **Corpus**. (e.g., `corpus/corpus.txt`)
- `--output_dir`: the output directory. (e.g., `dataset`)

Three files named `functions.txt`(edge information in paths), `samples.txt`(paths of each flaky/non-flaky tests) and `tokens.txt`(token dictionary) respectively will be generated under `output_dir`.

The processed corpus and dataset can be downloaded [here](https://pan.baidu.com/s/1WuxhiwHwOy0l0LhwxXZcqw?pwd=7mf3).

### Pretrained Embeddings
We utilize pretrained [**code2vec**](https://github.com/tech-srl/code2vec) to generate embeddings for tokens and functions. The commands are:
```shell
python3 embeddings/method2embeddings.py --input_file --code2vec_dir --output_dir
```
- `--input_file`: path to the `functions.txt` generated in step **Dataset**. (e.g., `dataset/functions.txt`)
- `--code2vec_dir`: directory of the code2vec project, notice that we modify code2vec slightly for compatibility, the pretrained model can be downloaded [here](https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz). (e.g., `code2vec`)
- `--output_dir`: the output directory where `methods_embeddings.txt` will be generated. (e.g., `embeddings`)

```shell
python3 embeddings/tokens2embeddings.py --file --vecs --output_dir
```
- `--file`: path to the `tokens.txt` generated in step **Dataset**. (e.g., `dataset/tokens.txt`)
- `--vecs`: path to the pretrained token embedding file, which can be downloaded [here](https://s3.amazonaws.com/code2vec/model/token_vecs.tar.gz). (e.g., `code2vec/token_vecs.txt`)
- `--output_dir`: the output directory where `tokens_embeddings.txt` will be generated. (e.g., `embeddings`)

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
python flaky_detect.py --predict_model --predict_mode True
```
- `--predict_model`: path to the pretrained model file. (e.g., `evaluation/10FoldCrossValidation/b64_encode384_pl20_pc100_hd128_tv256_lr001_10FoldCrossValidationcross_0.model`)

## Evaluation Results
We apply 10-fold validation for evaluating *Peeler* on the whole dataset, log in directory `logs`.
- RQ1: The pretrained models for each fold, evaluation results, as well as the comparison results are released in directory `evaluation/10FoldCrossValidation`.
- RQ2: The inner-file and inter-file ablation study results can be seen in `evaluation/ablation study`.

In RQ3, *Peeler* is utilized for detecting previously-unknown flaky tests in real-world java projects, *Peeler* predicts 65 out of 1,835 tests as flaky in total, we report 21 of them, among which 12 have been accepted by developers, the links to the issues are as follows:
- [https://github.com/google/guava/issues/5864](https://github.com/google/guava/issues/5864)
- [https://github.com/google/guava/issues/5865](https://github.com/google/guava/issues/5865)
- [https://github.com/google/guava/issues/5871](https://github.com/google/guava/issues/5871)
- [https://github.com/GoogleCloudPlatform/DataflowTemplates/issues/326](https://github.com/GoogleCloudPlatform/DataflowTemplates/issues/326)

For the remain 9 reports, the developers havn't yet reply to us, their links are:
- [https://github.com/alibaba/fastjson/issues/4004](https://github.com/alibaba/fastjson/issues/4004)
- [https://github.com/GoogleCloudPlatform/DataflowTemplates/issues/326](https://github.com/GoogleCloudPlatform/DataflowTemplates/issues/326)

Note that we reported 9 tests in the issue of project *DataflowTemplates*, one of them has been confirmed as flaky test and the remain 8 havn't been reproduced yet according to the developers.
