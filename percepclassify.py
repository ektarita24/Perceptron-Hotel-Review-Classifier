import sys
import codecs
import string

def get_parameters():
    global true_fake_weights, true_fake_bias, pos_neg_weights, pos_neg_bias
    perceptron_model = codecs.open(sys.argv[1], 'r', encoding='utf-8')

    line_no = 0
    for line in perceptron_model:
        if line_no == 1:
            true_fake_weights = eval(line)
        elif line_no == 3:
            true_fake_bias = eval(line)
        elif line_no == 5:
            pos_neg_weights = eval(line)
        elif line_no == 7:
            pos_neg_bias = eval(line)
        line_no += 1


def classify_review():
    nb_output_file = codecs.open("percepoutput.txt", "w", encoding='utf-8')

    for line in lines:
        true_fake_value = 0
        pos_neg_value = 0
        reviews = line.rstrip().split(" ", 1)
        reviews[1] = reviews[1].lower().replace("-", '').replace("'", '')
        for punc in string.punctuation:
            reviews[1] = reviews[1].replace(punc, ' ')

        feature_words = filter(None, reviews[1].split(" "))
        for feature_word in feature_words:
            if feature_word in true_fake_weights:
                true_fake_value += true_fake_weights[feature_word]

            if feature_word in pos_neg_weights:
                pos_neg_value += pos_neg_weights[feature_word]

        true_fake_value += true_fake_bias
        pos_neg_value += pos_neg_bias

        if true_fake_value > 0:
            nb_output_file.write(reviews[0] + " True ")
        else:
            nb_output_file.write(reviews[0] + " Fake ")

        if pos_neg_value > 0:
            nb_output_file.write("Pos\n")
        else:
            nb_output_file.write("Neg\n")

    nb_output_file.close()


'''Main'''
true_fake_weights = dict()
true_fake_bias = 0

pos_neg_weights = dict()
pos_neg_bias = 0

get_parameters()

dev_text_file = codecs.open(sys.argv[2], 'r', encoding='utf-8')

lines = dev_text_file.readlines()
classify_review()
