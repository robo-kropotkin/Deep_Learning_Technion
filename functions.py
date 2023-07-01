import re
from data.Contractions import contractions_dict

contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))

def expand_contractions(s, contractions_dict=contractions_dict):
  def replace(match):
    return contractions_dict[match.group(0)]
  return contractions_re.sub(replace, s)


def preprocessW(sentence):
    output = "[CLS] " + sentence.lower()
    output = expand_contractions(output)
    orginals = ["."]
    convert_to = [". [SEP]"]
    for index,word in enumerate(orginals):
        output = output.replace(word,convert_to[index])
    return output