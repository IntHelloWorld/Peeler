import traceback

from common import common
from extractor import Extractor

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
JAR_PATH = 'code2vec/JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar'


class InteractivePredictor:
    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        self.path_extractor = Extractor(config,
                                        jar_path=JAR_PATH,
                                        max_path_length=MAX_PATH_LENGTH,
                                        max_path_width=MAX_PATH_WIDTH)

    def read_file(self, input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self):
        input_filename = 'code2vec/Input.java'
        print('Starting interactive prediction...')
        try:
            predict_lines, hash_to_string_dict = self.path_extractor.extract_paths(
                input_filename)
        except ValueError as e:
            print(e)
            return None
        raw_prediction_results = self.model.predict(predict_lines)
        method_prediction_results = common.parse_prediction_results(
            raw_prediction_results, hash_to_string_dict,
            self.model.vocabs.target_vocab.special_words, topk=SHOW_TOP_CONTEXTS)
        for raw_prediction, method_prediction in zip(raw_prediction_results, method_prediction_results):
            print('Original name:\t' + method_prediction.original_name)
            # for name_prob_pair in method_prediction.predictions:
            #     print('\t(%f) predicted: %s' %
            #           (name_prob_pair['probability'], name_prob_pair['name']))
            # print('Attention:')
            # for attention_obj in method_prediction.attention_paths:
            #     print('%f\tcontext: %s,%s,%s' % (
            #         attention_obj['score'], attention_obj['token1'], attention_obj['path'], attention_obj['token2']))
            if self.config.EXPORT_CODE_VECTORS:
                return ' '.join(map(str, raw_prediction.code_vector))
