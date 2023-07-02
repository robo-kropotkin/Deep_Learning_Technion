import re
from data.Contractions import contractions_dict
from tqdm import tqdm


tqdm.pandas()
contractions_re = re.compile('(%s)'%'|'.join(contractions_dict.keys()))


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


def preprocessW(sentence, maxlength, tokenizer):
    output = "[CLS] " + sentence.lower()
    new_output = expand_contractions(output)
    while output != new_output:
        output = new_output
        new_output = expand_contractions(output)
    output = output.replace("\'s", " \'s")
    output = output.replace(".", ". [SEP]") # TODO: Ask if this is necessary, A:yes
    output = tokenizer.tokenize(output)
    output += ['[PAD]'] * (maxlength - len(output))
    return output
