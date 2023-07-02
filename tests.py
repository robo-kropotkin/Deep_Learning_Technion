import functions
import main


def functions_test():
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # define pre-trained tokenizer
    sentence = "ok this is a TEST you'll have to watch."
    sentence = functions.preprocessW(sentence, 400, tokenizer)
    print(sentence)


if __name__ == "__main__":
    functions_test()
