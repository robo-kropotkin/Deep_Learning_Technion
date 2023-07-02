import functions
import main

def functions_test():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # define pre-trained tokenizer
    sentence = "ok this is a TEST you'll have to watch."
    sentence = functions.preprocessW(sentence, 400, tokenizer)
    print(sentence)


def tokenizer_test():
    print(f"original sentence:\n{}\n preprocessed sentence:\n{}\n \
          tokenized sentence:\n{}\n".format(main.sample,main.x_train[0],main.tokenizer.convert_tokens_to_ids(main.x_train[0])))


if __name__ == "__main__":
    functions_test()
    tokenizer_test()
