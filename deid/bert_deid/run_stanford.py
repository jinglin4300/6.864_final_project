import pkgutil

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize

def run_stanfordNER(text):
    path_to_stanfordner = '/home/jingglin/research/6.864_final_project/deid/bert_deid/stanford-ner/'
    model_path = path_to_stanfordner + 'classifiers/english.all.3class.distsim.crf.ser.gz'
    jar_path = path_to_stanfordner + 'stanford-ner.jar'
    st = StanfordNERTagger(model_path, jar_path, encoding='utf-8')

    tokenized_text = word_tokenize(text)
    classified_tokens = st.tag(tokenized_text)
    classified_tokens_interest = [x for x in classified_tokens if x[1] != 'O']
    # print ('classified_tokens_interest', classified_tokens_interest)
    # for each, _ in classified_tokens_interest:
    #     offset = text.find(each)
    #     length = len(each)
    #     print ('token', each)
    #     print ('length', length)
    #     print ('offset', offset)
    lengths = []
    offsets = []
    
    # no PHI tokens identified
    if len(classified_tokens_interest) == 0:
        return lengths, offsets
    
    token_index = 0 
    for i in range(len(text)):
        token_length = len(classified_tokens_interest[token_index][0])
        # check if entity matched
        if text[i:i+token_length] == classified_tokens_interest[token_index][0]:
            lengths.append(token_length)
            offsets.append(i)
            token_index +=1
            if token_index == len(classified_tokens_interest):
                break
    return lengths, offsets

# text = '  While in France on 12/01/2009, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'
# lengths, offsets = run_stanfordNER(text)
# print ('lengths', lengths)
# print ('offsets', offsets)

