"""
Much of the work below is recycled from Joel Grus' 
O'Reilly book Data Science From Scratch.
"""
import re
import math
import random
from glob import glob
from collections import defaultdict, Counter
from stemming.porter import stem

class NaiveBayesClassifier:

    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []

    def train(self, training_set):
        #Count spam and non-spam message
        num_spams = len([is_spam
                         for message, is_spam in training_set
                         if is_spam])
        num_non_spams = len(training_set) - num_spams

        #Run training data through the classifier
        word_counts = count_words(training_set)
        self.word_probs = word_probabilities(word_counts,
                                             num_spams,
                                             num_non_spams,
                                             self.k)
    def classify(self, message):
        return spam_probability(self.word_probs, message)
                                             


def tokenize(message):
    
    """
    Finds all the words in message and returns them as a
    unique set
    """

    message = message.lower()
    all_words = re.findall("[a-z0-9']+", message)
    words = [stem(word) for word in all_words]
    return set(words)


def count_words(training_set):

    """
    training_set = pairs of (message, is_spam)
    
    returns: counts[word][is_spam] = {n_times}
    """
    counts = defaultdict(lambda: [0,0])

    for message, is_spam in training_set:

        for word in tokenize(message):

            #word = stem(word)
            counts[word][0 if is_spam else 1] += 1

    return counts


def word_probabilities(counts, total_spams, total_non_spams, k=0.5):

    """
    Turns the word counts into a list of triplets
    for easier manipulation:
    [word, p(word|spam), p(word|~spam)]
    """
    return [(w,
             (spam+k) / (total_spams + 2*k),
             (non_spam+k) / (total_non_spams + 2*k))
            for w, (spam, non_spam) in counts.iteritems()]


def spam_probability(word_probs, message):

    """
    Returns conditional probability message is spam
    based on incidence of words as derived from the training
    set present in word_probs
    """

    message_words = tokenize(message)
    log_prob_if_spam = log_prob_if_not_spam = 0.0

    #Iterate through the training set's words
    for word, prob_if_spam, prob_if_not_spam in word_probs:

        # If training set word appears in the message,
        # add the ln(prob) of seeing it
        if word in message_words:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_not_spam += math.log(prob_if_not_spam)

        # If training set word *not* in the message,
        # add the probability of *not* seeing it
        else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

    # Retun the probability that message is spam
    prob_if_spam = math.exp(log_prob_if_spam)
    prob_if_not_spam = math.exp(log_prob_if_not_spam)
    return prob_if_spam / (prob_if_spam + prob_if_not_spam)

def split_data(data, prob):
    """
    Split data into a training and validation/test
    set, where prob tells what percent of the data to
    put in the training set
    """
    results = [],[]
    for row in data:
        results[0 if random.random() < prob else 1].append(row)

    return results
    
def testSet():

    # Grab the spam data files
    path = r'spam_data/*/*'
    data = []

    # Go through all the files to begin constructing the training
    # set
    for fn in glob(path):
        # Check if the message comes from a spam folder
        is_spam = "ham" not in fn

        with open(fn, 'r') as file:
            for line in file:

                if line.startswith("Subject:"):
                # Remove the Subject header and
                # leave the title
                    subject = re.sub(r'^Subject: ', '', line).strip()
                    data.append((subject, is_spam))

    # Now split the data into training and validation sets
    random.seed(0)
    train_data, valid_data = split_data(data, 0.75)

    # Run the training set through the algorithm
    classifier = NaiveBayesClassifier()
    classifier.train(train_data)

    classified = [(subject, is_spam, classifier.classify(subject))
                  for subject, is_spam in valid_data]

    # Assume p > 0.5 corresponds to spam
    count = Counter((is_spam, spam_probability > 0.5)
                    for _, is_spam, spam_probability in classified)

    return count

if __name__ == "__main__":

    print "Running test..."
    count = testSet()
    print "   ", count[(False,False)], "Non-spam identified as Non-spam (True Negative)"
    print "   ", count[(True,True)], " Spam identified as Spam (True Positive)"
    print "   ", count[(False,True)], " Non-spam identified as Spam (False Positive)"
    print "   ", count[(True,False)], " Spam identified as Non-spam (False Negative)"
