def preprocessW(sentence):
    output = "[CLS] " + sentence.lower()
    orginals = ["."]
    convert_to = [". [SEP]"]
    for index,word in enumerate(orginals):
        output = output.replace(word,convert_to[index])
    return output

print(preprocessW('bert base uncased.'))