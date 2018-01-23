import numpy as np
from nltk.tree import *
from sklearn.decomposition import NMF


def get_context_embedding(model, parsed_sentence):

    ''' Returns the vectors for the sentence, phrase and word

    :param model: word2vec model
    :param parsed_sentence: parsed tree of the text.
    :return text embedding :rtype list: embedding of the text
    :return final_phrases :rtype list: words and phrases in the text
    :return texts_vec :rtype list: vectors for words and phrases
    :return final_phrases_name :rtype list: pos tags of the word and phrase tags

    '''

    def normalize_vector(temp_vec):
        normalized_vector = (temp_vec - np.amin(temp_vec)) / (np.amax(temp_vec) - np.amin(temp_vec))
        return normalized_vector

    def term_to_vec(term):
        try:
            temp_vec = model[term]
            temp_vec = normalize_vector(temp_vec)
        except KeyError:
            temp_vec = np.random.rand(model.vector_size, )
        return temp_vec.tolist()

    def terms_to_vec(sentence_matrix):
        sentence_matrix = sentence_matrix.transpose()
        nmf = NMF(n_components=1, init='random', random_state=0, max_iter=200)
        decomposed_vec = nmf.fit_transform(sentence_matrix)
        decomposed_vec = [thing[0] for thing in decomposed_vec.tolist()]
        return decomposed_vec

    def get_word_phrase_sentence_vec():
        full_tree = Tree.fromstring(parsed_sentence)
        phrases = []
        matrix = []
        phrases_name = []
        temp_phrases = []
        for Height in range(2, full_tree.height()):
            for sub_tree in full_tree.subtrees():
                temp_phrases = []
                phrase = ''
                if (sub_tree.height() == Height) and (sub_tree.height() == 2):
                    phrase = ' '.join([thing for thing in sub_tree.flatten()])
                    for thing in phrase.split():
                        vector = term_to_vec(thing)
                        matrix.append(vector)
                        phrases.append(thing)
                        phrases_name.append(sub_tree.label())
                elif (sub_tree.height() == Height) and (sub_tree.height() == 3):
                    temp_matrix = []
                    for thing in sub_tree.flatten():
                        temp_matrix.append(matrix[phrases.index(thing)])
                    temp_matrix = np.matrix(temp_matrix)
                    vector = terms_to_vec(temp_matrix)

                    phrase = ' '.join([thing for thing in sub_tree.flatten()])
                    phrases.append(phrase)
                    matrix.append(vector)
                    phrases_name.append(sub_tree.label())
                elif (sub_tree.height() == Height) and (sub_tree.height() > 3):
                    phrase = ' '.join([thing for thing in sub_tree.flatten()])
                    temp_phrases.append(phrase)
                    phrase1 = phrase
                    temp_matrix = []
                    position = []
                    for temp_var in range(len(phrases) - 1, -1, -1):
                        if len(phrases[temp_var].split()) > 1:
                            if phrases[temp_var] in phrase1:
                                position.append(phrase1.find(phrases[temp_var]))
                                temp_matrix.append(matrix[phrases.index(phrases[temp_var])])
                                phrase1 = phrase1.replace(phrases[temp_var], '')
                        if len(phrases[temp_var].split()) == 1:
                            temp_phrase1 = phrases[temp_var] + ' '
                            temp_phrase2 = ' ' + phrases[temp_var]
                            if temp_phrase1 in phrase1:
                                position.append(phrase1.find(phrases[temp_var]))
                                temp_matrix.append(matrix[phrases.index(phrases[temp_var])])
                                phrase1 = phrase1.replace(temp_phrase1, '')
                            elif temp_phrase2 in phrase1:
                                position.append(phrase1.find(phrases[temp_var]))
                                temp_matrix.append(matrix[phrases.index(phrases[temp_var])])
                                phrase1 = phrase1.replace(temp_phrase2, '')
                            elif (phrases[temp_var] in phrase1) and (len(phrases[temp_var]) == len(phrase1)):
                                position.append(phrase1.find(phrases[temp_var]))
                                temp_matrix.append(matrix[phrases.index(phrases[temp_var])])
                                phrase1 = phrase1.replace(phrases[temp_var], '')
                            else:
                                pass
                    phrases.append(phrase)
                    phrases_name.append(sub_tree.label())
                    temp_matrix_1 = []
                    for i in range(0, len(position)):
                        temp_var_1 = position.index(min(position))
                        temp_matrix_1.append(temp_matrix[temp_var_1])
                        position.remove(min(position))
                    temp_matrix = np.matrix(temp_matrix_1)
                    vector = np.asarray(np.sum(temp_matrix, axis=0))
                    matrix.append((vector.tolist())[0])
                else:
                    pass
            phrases = phrases + temp_phrases
        return phrases, matrix, phrases_name

    final_phrases, texts_vec, final_phrases_name = get_word_phrase_sentence_vec()
    sentence_embedding = texts_vec[len(texts_vec) - 1]
    return sentence_embedding, final_phrases, texts_vec, final_phrases_name
