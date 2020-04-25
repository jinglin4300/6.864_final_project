import pydeid
import pkgutil
from pydeid.annotation import Document, EntityType
from pydeid.annotators import Pattern

from pydeid.annotators import _patterns

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

# load all modules on path
pkg = 'pydeid.annotators._patterns'
_PATTERN_NAMES = [name for _, name, _ in pkgutil.iter_modules(
    _patterns.__path__
)]

_STANFORD_PATTERN_NAMES=["PERSON", "ORGANIZATION", "LOCATION"]

def find_ner_location(pattern_name, pattern_label, text):
    
    path_to_stanfordner = '/home/jingglin/research/6.864_final_project/deid/bert_deid/stanford-ner/'
    model_path = path_to_stanfordner + 'classifiers/english.all.3class.distsim.crf.ser.gz'
    jar_path = path_to_stanfordner + 'stanford-ner.jar'
    st = StanfordNERTagger(model_path, jar_path, encoding='utf-8')
    
    tokenized_text = word_tokenize(text)
    classified_text = st.tag(tokenized_text)
    classified_text_interest = [x for x in classified_text if x[1] == pattern_name]
    ner_loc = [0]*len(text)
    
    if len(classified_text_interest) == 0:
        return ner_loc
    
    word_index = 0 
    for i in range(len(text)):
        word_length = len(classified_text_interest[word_index][0])
        if text[i:i+word_length] == classified_text_interest[word_index][0]:
            ner_loc[i:i+word_length] = [pattern_label]*word_length
            word_index +=1
            if word_index == len(classified_text_interest):
                break
    return ner_loc
        

def find_phi_location(pattern_name, pattern_label, text, stanfordNER = False):

    if pattern_name is None:
        return None
    if stanfordNER:
        pattern_name = pattern_name.upper()
        if pattern_name not in _STANFORD_PATTERN_NAMES:
            raise ValueError("Invalid pattern argument for stanford NER")
        return find_ner_location(pattern_name, pattern_label, text)
    if pattern_name.lower() not in _PATTERN_NAMES:
        raise ValueError("Invalid pattern argument")

    doc = Document(text)

    # find PHI with specific pydeid pattern
    entity_types = [EntityType(pattern_name)]
    modules = [pattern_name]
    model = Pattern(modules, entity_types)
    txt_annotated = model.annotate(doc)

    # mark location of detected phi with 1s and rest with 0s
    phi_loc = [0] * len(text)
    for ann in txt_annotated.annotations:
        start, end = ann.start, ann.end
        phi_loc[start:end] = [pattern_label] * (end-start)
    
    return phi_loc

def create_extra_feature_vector(phi_loc, input_offsets, input_lengths, token_sw, max_seq_length=128):
    # transform feature to match with BERT tokenization offset
    feature_vector = []
    for i in range(len(input_offsets)):
        start = input_offsets[i]
        # offset uses negative to indicate special token padding
        if start >= 0: # valid input token
            stop = start + input_lengths[i]
            if not token_sw[i]:
                # token is assigned with most occured label for correspoinding characters
                feature_vector.append(max(phi_loc[start:stop], key=list(phi_loc[start:stop]).count))
            else:
                # similarily as BERT, aggregate subword token label to the very first token
                feature_vector.append(0)

    # adds [CLS] at front
    feature_vector = [0] + feature_vector
    # padd rest with zeros
    feature_vector += [0] * (max_seq_length - len(feature_vector))

    return feature_vector