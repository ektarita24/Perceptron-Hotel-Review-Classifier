import string
import sys
import collections
import codecs


def vanilla_perceptron_model():
    vanilla_selected_features = dict()

    true_fake_weights = dict()
    pos_neg_weights = dict()
    true_fake_bias = 0
    pos_neg_bias = 0

    true_fake_label = dict()
    pos_neg_label = dict()

    for line in lines:
        words = line.rstrip().split(" ", 3)
        words[3] = words[3].lower().replace("-", '').replace("'", '')
        for punc in string.punctuation:
            words[3] = words[3].replace(punc, ' ')

        if words[1] == "True":
            true_fake_label[words[0]] = 1
        else:
            true_fake_label[words[0]] = -1

        if words[2] == "Pos":
            pos_neg_label[words[0]] = 1
        else:
            pos_neg_label[words[0]] = -1

        w = filter(None, words[3].split(" "))
        features = dict()
        for word in w:
            if word not in stopwords:
                if word in features:
                    features[word] += 1
                else:
                    features[word] = 1
                if word not in true_fake_weights:
                    true_fake_weights[word] = 0
                    pos_neg_weights[word] = 0
        vanilla_selected_features[words[0]] = features

    for i in range(0, 25):
        for key in vanilla_selected_features:
            feature_words = vanilla_selected_features[key]

            '''True/Fake'''
            true_fake_value = 0
            for word in feature_words:
                true_fake_value += true_fake_weights[word] * feature_words[word]
            true_fake_value += true_fake_bias

            if true_fake_value * true_fake_label[key] <= 0:
                for word in feature_words:
                    true_fake_weights[word] += feature_words[word] * true_fake_label[key]
                true_fake_bias += true_fake_label[key]

            '''Pos/Neg'''
            pos_neg_value = 0
            for word in feature_words:
                pos_neg_value += pos_neg_weights[word] * feature_words[word]
            pos_neg_value += pos_neg_bias

            if pos_neg_value * pos_neg_label[key] <= 0:
                for word in feature_words:
                    pos_neg_weights[word] += feature_words[word] * pos_neg_label[key]
                pos_neg_bias += pos_neg_label[key]

    vanilla_model_file = codecs.open("vanillamodel.txt", "w", encoding='utf-8')
    vanilla_model_file.write("true_fake_weights\n" + str(true_fake_weights) + "\n")
    vanilla_model_file.write("true_fake_bias\n" + str(true_fake_bias) + "\n")
    vanilla_model_file.write("pos_neg_weights\n" + str(pos_neg_weights) + "\n")
    vanilla_model_file.write("pos_neg_bias\n" + str(pos_neg_bias))
    vanilla_model_file.close()


def average_perceptron_model():
    averaged_selected_features = collections.OrderedDict()

    true_fake_weights = dict()
    pos_neg_weights = dict()

    true_fake_cached_weights = dict()
    pos_neg_cached_weights = dict()

    true_fake_bias = 0
    pos_neg_bias = 0

    true_fake_cached_bias = 0
    pos_neg_cached_bias = 0

    true_fake_label = dict()
    pos_neg_label = dict()

    for line in lines:
        words = line.rstrip().split(" ", 3)
        words[3] = words[3].lower().replace("-", '').replace("'", '')
        for punc in string.punctuation:
            words[3] = words[3].replace(punc, ' ')

        if words[1] == "True":
            true_fake_label[words[0]] = 1
        else:
            true_fake_label[words[0]] = -1

        if words[2] == "Pos":
            pos_neg_label[words[0]] = 1
        else:
            pos_neg_label[words[0]] = -1

        w = filter(None, words[3].split(" "))
        features = dict()
        for word in w:
            if word not in stopwords:
                if word in features:
                    features[word] += 1
                else:
                    features[word] = 1
                if word not in true_fake_weights:
                    true_fake_weights[word] = 0
                    true_fake_cached_weights[word] = 0
                    pos_neg_weights[word] = 0
                    pos_neg_cached_weights[word] = 0
        averaged_selected_features[words[0]] = features

    c = 1
    for i in range(0, 25):
        for key in averaged_selected_features:
            feature_words = averaged_selected_features[key]

            '''True/Fake'''
            true_fake_value = 0
            for word in feature_words:
                true_fake_value += true_fake_weights[word] * feature_words[word]
            true_fake_value += true_fake_bias

            if true_fake_value * true_fake_label[key] <= 0:
                for word in feature_words:
                    true_fake_weights[word] += feature_words[word] * true_fake_label[key]
                    true_fake_cached_weights[word] += feature_words[word] * true_fake_label[key] * c
                true_fake_bias += true_fake_label[key]
                true_fake_cached_bias += true_fake_label[key] * c

            '''Pos/Neg'''
            pos_neg_value = 0
            for word in feature_words:
                pos_neg_value += pos_neg_weights[word] * feature_words[word]
            pos_neg_value += pos_neg_bias

            if pos_neg_value * pos_neg_label[key] <= 0:
                for word in feature_words:
                    pos_neg_weights[word] += feature_words[word] * pos_neg_label[key]
                    pos_neg_cached_weights[word] += feature_words[word] * pos_neg_label[key] * c
                pos_neg_bias += pos_neg_label[key]
                pos_neg_cached_bias += pos_neg_label[key] * c

            c = c + 1

    for word in true_fake_weights:
        true_fake_cached_weights[word] = true_fake_weights[word] - true_fake_cached_weights[word] / float(c)
        pos_neg_cached_weights[word] = pos_neg_weights[word] - pos_neg_cached_weights[word] / float(c)

    true_fake_cached_bias = true_fake_bias - true_fake_cached_bias / float(c)
    pos_neg_cached_bias = pos_neg_bias - pos_neg_cached_bias / float(c)

    averaged_model_file = codecs.open("averagedmodel.txt", "w", encoding='utf-8')
    averaged_model_file.write("true_fake_weights\n" + str(true_fake_cached_weights) + "\n")
    averaged_model_file.write("true_fake_bias\n" + str(true_fake_cached_bias) + "\n")
    averaged_model_file.write("pos_neg_weights\n" + str(pos_neg_cached_weights) + "\n")
    averaged_model_file.write("pos_neg_bias\n" + str(pos_neg_cached_bias))
    averaged_model_file.close()


'''Main'''
training_data = codecs.open(sys.argv[1], 'r', encoding='utf-8')

lines = training_data.readlines()
stopwords = ['a', 'about', 'above', 'after', 'again', 'against', 'am', 'an', 'and', 'are', 'as', 'at', 'be', 'because',
             'been', 'before', 'being', 'below', 'between', 'but', 'by', 'can', 'did', 'do', 'does', 'doing', 'down',
             'during', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself',
             'him', 'himself', 'his', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'my',
             'myself', 'now', 'of', 'off', 'on', 'once', 'or', 'our', 'ours', 'ourselves', 'out', 'over', 'she', 'should', 'so',
             'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this',
             'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when',
             'where', 'which', 'while', 'who', 'whom', 'will', 'with', 'you', 'your', 'yours', 'yourself']

vanilla_perceptron_model()
average_perceptron_model()
