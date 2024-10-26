# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        """

        question -- The JSON object of the original question, you can extract metadata from this such as the category

        run -- The subset of the question that the guesser made a guess on

        guess -- The guess created by the guesser

        guess_history -- Previous guesses (needs to be enabled via command line argument)

        other_guesses -- All guesses for this run
        """


        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        # How many characters long is the question?

        yield ("question char", log(1 + len(run)))

        # How many words long is the question?
        yield ("run word", log(1 + len(run.split())))

        # How many characters long is the guess?
        if guess is None or guess=="":  
            yield ("guess char", -1)         
        else:                           
            yield ("guess char", log(1 + len(guess)))
            
class FrequencyFeature(Feature):
    # How many pages appear in a training set 
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.counts = Counter()
        self.normalize = normalize_answer

    def add_training(self, question_source):
        import json
        import gzip
        if 'json.gz' in question_source:
            with gzip.open(question_source) as infile:
                questions = json.load(infile)
        else:
            with open(question_source) as infile:
                questions = json.load(infile)
        for ii in questions:
            self.counts[self.normalize(ii["page"])] += 1

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ("guess", log(1 + self.counts[self.normalize(guess)]))
      

class CategoryFeature(Feature):
    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history, other_guesses=None):
        yield ("category", question["category"])
        
class WikiFeature(Feature):

    def __init__(self, name):
        self.name = name
        self.dict = {}

    def add_training(self, question_source):
        import json
        with open(question_source, 'r') as file:
            json_content = file.read()
        pages = json.loads(json_content)
        for ii in pages:
            if ii["page"] in self.dict:
                self.dict[ii["page"]] += ii["text"]
            else:
                self.dict[ii["page"]] = ii["text"]

    def __call__(self, question, run, guess, guess_history):
        #print(self.dict.keys())
        guess = guess.replace(" ", "_")
        if guess in self.dict:
            words1 = self.dict[guess].split()
            words2 = run.split()

            set1 = set(words1)
            set2 = set(words2)
            overlapped_words = len(set1.intersection(set2))
            if (len(run.split())) != 0:        
                yield ("guess", overlapped_words/(len(run.split())))
                yield("found", 1)
            else:
                yield ("guess", 0)
                yield("found", 0.5)
        elif guess+'s' in self.dict:
            words1 = self.dict[guess+'s'].split()
            words2 = run.split()

            set1 = set(words1)
            set2 = set(words2)
            overlapped_words = len(set1.intersection(set2))

            if (len(run.split())) != 0:        
                yield ("guess", overlapped_words/(len(run.split())))
                yield("found", 1)
            else:
                yield ("guess", 0)
                yield("found", .5)
        else:
            overlapped_words = 0
            yield("guess", overlapped_words)
            yield("found", 0)
        
class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """
    def __call__(self, question, run, guess, guess_history, guesses):
        yield ('true', len(guess) == 0)


class GuessCapitalsFeature(Feature):
    """
    Capital letters in guess
    """
    def __call__(self, question, run, guess ,guess_history, guesses):
        yield ('true', log(sum(i.isupper() for i in guess) + 1))
        
class GuessInRunFeature(Feature):
    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history,guesses):
        guess_position = run.lower().find(guess.lower())
        if guess_position == -1:
            normalized_position = -1
        else:
            normalized_position = 1
        yield ("GuessInRun", normalized_position)

class CategoryFeature(Feature):
    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history,guesses):
        category = question["category"]
        yield("Category", category)
    
class AverageWordLengthFeature(Feature):
    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history,guesses):
        words = run.split()
        if not words:
            yield ("AvgWordLength", 0)
        else:
            avg_word_length = sum(len(word) for word in words) / len(words)
            yield ("AvgWordLength", log(avg_word_length + 1))
            
class DifficultyFeature(Feature):
    def __call__(self, question, run, guess, guess_history,guesses):
        if question["difficulty"] == "College":
            yield ("College", True)
        else:
            yield ("College", False)

class YearFeature(Feature):
    def __call__(self, question, run, guess, guess_history,guesses):
        yield ("", (question["year"] - 1997)/18)

class HasOfFeature(Feature):
    def __call__(self, question, run, guess, guess_history,guesses):
        yield ("", "of" in guess)

class PronounsFeature(Feature):
    def __call__(self, question, run, guess, guess_history,guesses):
        close = 0
        distant = 0
        personal = 0
        for word in run.split():
            if word.lower() == "this" or word.lower() == "these":
                close += 1
            elif word.lower() == "that" or word.lower() == "their":
                distant += 1
            elif word.lower() == "he" or word.lower() == "she" or word.lower() == "it" or word.lower() == "they":
                personal += 1
        yield ("close", log(1 + close))
        yield ("distant", log(1 + distant)) 
        yield ("personal", log(1 + personal)) 
        
class PluralityFeature(Feature):
    def __call__(self, question, run, guess, guess_history,guesses):
        yield ("a", int("these" in run.split()) == (guess and guess[-1] == 's'))

class SpaceFeature(Feature):
    def __call__(self, question, run, guess, guess_history,guesses):
        yield ("", " " in guess)

class HasDisambiguationFeature():
    def __init__(self, name):
        from eval import normalize_answer
        self.name = name
        self.normalize = normalize_answer

    def __call__(self, question, run, guess, guess_history,guesses):
        if "(" in guess and ")" in guess:
            disambiguation = guess[guess.index("(") + 1: guess.index(")")].lower()
            yield ("", (disambiguation in run.lower() or disambiguation in question["tournament"].lower()  or disambiguation in question["subcategory"].lower()))
        else:
            yield ("", False)


if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from parameters import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    guesser_params = add_guesser_params(parser)
    buzzer_params = add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags, guesser_params)
    buzzer = load_buzzer(flags, buzzer_params)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
