
import os
import spacy
from allennlp.commands.train import train_model
from allennlp.common import Params
from allennlp.modules import ConditionalRandomField
from allennlp_models.tagging.models.crf_tagger import CrfTagger
from datetime import datetime
from reader import CsvReader

nlp = spacy.load('en_core_web_sm') 
nlp.max_length = 3278900
if __name__ == "__main__":
    params_file = "extraction_distillbert.json"
    params_file = "extraction_glove.json"
    params = Params.from_file(params_file)
    ser_file = "models/model"+str(datetime.now()).replace(" ", "-").replace(":","-").split(".")[0]+"/"
    if not os.path.isdir(ser_file):
        os.makedirs(ser_file)
    model = train_model(params, ser_file)
    print(ser_file)