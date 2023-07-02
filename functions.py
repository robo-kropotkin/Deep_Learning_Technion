import re
from data.Contractions import contractions_dict

contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, s)


def preprocessW(sentence,maxlength,tokenizer):
    output = "[CLS] " + sentence.lower()
    while (output != expand_contractions(output)):
      output = expand_contractions(output)
    # orginals = ["."]
    # convert_to = [". [SEP]"]
    # for index,word in enumerate(orginals):
    #     output = output.replace(word, convert_to[index])
    # output = tokenizer.tokenize(output)
    # for i in range(len(output), maxlength):
    #   output.append('[PAD]')
    return output

if __name__ == "__main__":
   from transformers import BertTokenizer
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # define pre-trained tokenizer
   sentence = "ok this is a TEST you'll've to watch."
   sentence = preprocessW(sentence, 400, tokenizer)
   print(sentence)