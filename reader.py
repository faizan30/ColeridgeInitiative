import csv
import json

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (SequenceLabelField,
                                  TextField, MetadataField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, SpacyTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer, token_indexer
import spacy
from typing import Dict

@DatasetReader.register('data_reader')
class CsvReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer= None,
            token_indexers: Dict[str, TokenIndexer]= None):
        super().__init__()
        self.tokenizer = tokenizer or SpacyTokenizer()
        self.token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}

    def _read(self, file_path: str):
        dir_name = "/".join(file_path.split("/")[:-1])+"/train/"
        new_examples = []
        instances = []
        with open(file_path, 'r') as inp:
            reader = csv.reader(inp, delimiter=",")
            # max_instances = 10000000
            max_instances = 100
            # if "test" in file_path:
            #     max_instances = max_instances*0.1
            # if "validation" in file_path:
            #     max_instances = max_instances*0.2
            for i, example in enumerate(reader):
                if i>=max_instances:
                    break
                clean_label = example[-1].lower()
                if example[0] == 'Id':
                    print("Header", example)
                    continue
                paper_id = example[0]
                try:
                    with open(dir_name+paper_id+".json") as fp:
                        content_list = json.load(fp)
                        contains_content = 0
                        for content in content_list:
                            if clean_label in content['text'].lower():
                                # print(clean_label)
                                contains_content += 1
                                text = content['text'].lower()
                                clean_label = clean_label.lower()
                                example.extend(
                                    [content['section_title'], text])
                                new_examples.extend([example])
                                instances.append(self.text_to_instance(text, clean_label))
                                print("Instances completed: %s\r" % i, end="")
                except FileNotFoundError as _:
                    print(dir_name+paper_id+".json" + " not found")
                    pass
                except ValueError as _:
                    print("spacy max length error: SKIP")
                    pass
        return instances
    def text_to_instance(self, text, clean_label='', section_title=''):
        fields = {}
        tokens = self.tokenizer.tokenize(text)
        fields['tokens'] = TextField(tokens, self.token_indexers)
        if section_title:
            print("WRONG")
            pass
        if clean_label:
            label_tokens = self.tokenizer.tokenize(clean_label)
            label_tokens = [lt.text for lt in label_tokens]
            tokens_text = [token.text for token in tokens]
            matched_index = self.get_tag_index(tokens_text, label_tokens)
            tags = self.assign_tags(tokens, matched_index)
            fields['tags'] = SequenceLabelField(tags, fields['tokens'])
        return Instance(fields)

    @staticmethod
    def get_tag_index(tokens, label_tokens):
        matched_index = []
        temp_match = []
        i = 0
        for j in range(0, (len(tokens))):
            if i < len(label_tokens):
                if label_tokens[i] == tokens[j]:
                    temp_match.append(j)
                    i += 1
                else:
                    i = 0
                    temp_match = []
            if len(temp_match) == len(label_tokens):
                matched_index.append(temp_match)
                i = 0
                temp_match = []

        return matched_index

    @staticmethod
    def assign_tags(tokens, matched_index):
        tags = ['O' for tk in tokens]
        for mt in matched_index:
            for i, m in enumerate(mt):
                if i == 0:
                    tags[m] = "B-tag"
                else:
                    tags[m] = "I-tag"

        return tags


if __name__ == "__main__":
    # text = ["machine", "learning", "is", "cool", "i", "like", "machines","lets","learn", "machine", "learning"]
    # label = ["machine", "learning"]
    # text = ['this', 'study', 'used', 'data', 'from', 'the', 'national', 'education', 'longitudinal', 'study', '(', 'nels:88', ')', 'to', 'examine', 'the', 'effects', 'of', 'dual', 'enrollment', 'programs', 'for', 'high', 'school', 'students', 'on', 'college', 'degree', 'attainment', '.', 'the', 'study', 'also', 'reported', 'whether', 'the', 'impacts', 'of', 'dual', 'enrollment', 'programs', 'were', 'different', 'for', 'first', 'generation', 'college', 'students', 'versus', 'students', 'whose', 'parents', 'had', 'attended', 'at', 'least', 'some', 'college', '.', 'in', 'addition', ',', 'a', 'supplemental', 'analysis', 'reports', 'on', 'the', 'impact', 'of', 'different', 'amounts', 'of', 'dual', 'enrollment', 'course', '-', 'taking', 'and', 'college', 'degree', 'attainment', '.', 'dual', 'enrollment', 'programs', 'offer', 'college', '-', 'level', 'learning', 'experiences', 'for', 'high', 'school', 'students', '.', 'the', 'programs', 'offer', 'college', 'courses', 'and/or', 'the', 'opportunity', 'to', 'earn', 'college', 'credits', 'for', 'students', 'while', 'still', 'in', 'high', 'school', '.', 'the', 'intervention', 'group', 'in', 'the', 'study', 'was', 'comprised', 'of', 'nels', 'participants', 'who', 'attended', 'a', 'postsecondary', 'school', 'and', 'who', 'participated', 'in', 'a', 'dual', 'enrollment', 'program', 'while', 'in', 'high', 'school', '(', 'n', '=', '880', ')', '.', 'the', 'study', 'author', 'used', 'propensity', 'score', 'matching', 'methods', 'to', 'create', 'a', 'comparison', 'group', 'of', 'nels', 'participants', 'who', 'also', 'attended', 'a', 'postsecondary', 'school', 'but', 'who', 'did', 'not', 'participate', 'in', 'a', 'dual', 'enrollment', 'program', 'in', 'high', 'school', '(', 'n', '=', '7,920', ')', '.']
    # label = ["national", "education", "longitudinal", "study"]
    # matched_index = CsvReader.get_tag_index(text, label)
    # tags = CsvReader.assign_tags(text, matched_index)
    # print(tags)

    file_path = ".data/coleridgeinitiative-show-us-the-data/train.csv"
    dir_name = ".data/coleridgeinitiative-show-us-the-data/train/"
    reader = CsvReader()
    instances = reader.read(file_path, dir_name)

    for i, ins in enumerate(instances):
        if i > 10:
            break
        print(ins)
