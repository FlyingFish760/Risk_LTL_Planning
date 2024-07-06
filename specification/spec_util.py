import itertools
import re


def parse_expression(expression):
    # Replace logical operators with Python equivalents
    expression = expression.replace("&", " and ")
    expression = expression.replace("|", " or ")
    expression = expression.replace("~", " not ")
    return expression


def get_word_combinations(words):
    # Generate all possible combinations of the given words with true/false
    return list(itertools.product([True, False], repeat=len(words)))


def evaluate_condition(condition, words, combination):
    expression = parse_expression(condition)
    # Create a dictionary to map words to their boolean values
    word_dict = dict(zip(words, combination))
    # Replace words in the expression with their boolean values
    for word, value in word_dict.items():
        expression = re.sub(r'\b' + word + r'\b', str(value), expression)
    # Evaluate the final expression
    return eval(expression)


def filter_combinations(condition, words):
    word_combinations = get_word_combinations(words)
    satisfying_combinations = []

    for combination in word_combinations:
        if evaluate_condition(condition, words, combination):
            satisfying_combinations.append(combination)

    return satisfying_combinations



if __name__ == "__main__":
    # Define the words and the logical condition
    words = ['o', 'c', 'g']
    condition = "~o | (~c & ~g)"
    # Get satisfying combinations
    satisfying_combinations = filter_combinations(condition, words)
    # Print the satisfying combinations
    print("The words and their satisfying combinations are:")
    for combination in satisfying_combinations:
        result = " & ".join(f"{words[i]}={'True' if combination[i] else 'False'}" for i in range(len(words)))
        print(result)


