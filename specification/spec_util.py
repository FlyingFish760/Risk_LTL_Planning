import itertools
import re


def parse_expression(expression):
    # Replace logical operators with Python equivalents
    expression = expression.replace("&", " and ")
    expression = expression.replace("|", " or ")
    expression = expression.replace("~", " not ")
    return expression


def get_word_combinations(AP_set):
    # Generate all possible combinations of the given words with true/false
    return list(itertools.product([True, False], repeat=len(AP_set)))


def evaluate_condition(condition, AP_set, letter):
    expression = parse_expression(condition)
    # Create a dictionary to map words to their boolean values
    letter_dict = dict(zip(AP_set, letter))
    # Replace words in the expression with their boolean values
    for AP, value in letter_dict.items():
        expression = re.sub(r'\b' + AP + r'\b', str(value), expression)
    # Evaluate the final expression
    return eval(expression)


def filter_combinations(condition, AP_set):
    alphabet = get_word_combinations(AP_set)
    satisfying_letter = []

    for letter in alphabet:
        if evaluate_condition(condition, AP_set, letter):
            satisfying_letter.append(letter)

    return satisfying_letter

def letters_to_flags(ap_set, letters):
    # Create a tuple based on the presence of each letter in ap_set in the string 'letters'
    return tuple(letter in letters for letter in ap_set)


if __name__ == "__main__":
    # Define the words and the logical condition
    AP_set = ['o', 'c', 'g']
    condition = "~o | (~c & ~g)"
    # Get satisfying combinations
    satisfying_combinations = filter_combinations(condition, AP_set)
    # Print the satisfying combinations
    print("The words and their satisfying combinations are:")
    for combination in satisfying_combinations:
        result = " & ".join(f"{AP_set[i]}={'True' if combination[i] else 'False'}" for i in range(len(AP_set)))
        print(result)

    # Example usage
    ap_set = ['o', 'c', 'g']
    letters = 'go'
    result = letters_to_flags(ap_set, letters)
    print(result)  # Output will be (True, False, True)



