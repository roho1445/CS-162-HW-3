import evaluate

# Load the sacrebleu metric
bleu = evaluate.load("sacrebleu")

# Example: Replace these with your actual translations and references
predictions = [
    "The city where Michelle was born has a population of 145,826. What is the value of the 5 in the number 145,826?",
    "Olivia uses the rule 'add 11' to create the following number sequence. Which pattern for the number sequence 10, 21, 32, 43, 54 is correct?",
    "A total of 30 players are playing basketball in a park. Each team has exactly 5 players. Which statement correctly explains how to find the number of teams needed?",
    "A store sells 107 different colors. In the storage, there are 25 bottles of each color. The total number of color bottles in storage can be found using the expression 107 × 25. How many color bottles are there in storage?",
    "Which expression is equal to 5 × 9?",
    "The population growth rate is found by subtracting.",
    "In the third stage of demographic transition, which of the following is true?",
    "In the United States, regarding the services provided by local governments, which of the following statements is incorrect?",
    "The practice of hiring an external third-party service provider to carry out operations is called outsourcing.",
    "Which of the following items is an example of a non-material culture?"
]

references = [
    ["The population of the city where Michelle was born is 145,826. What is the value of the 5 in the number 145,826?"],
    ['Olivia used the rule "Add 11" to create the number pattern shown below. 10, 21, 32, 43, 54 Which statement about the number pattern is true?'],
    ["A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?"],
    ["A store sells 107 different colors of paint. They have 25 cans of each color in storage. The number of cans of paint the store has in storage can be found using the expression below. 107 × 25. How many cans of paint does the store have in storage?"],
    ["Which expression is equivalent to 5 x 9?"],
    ["The rate of natural increase of a population is found by subtracting the"],
    ["During the third stage of the demographic transition model, which of the following is true?"],
    ["Which of the following statements is NOT accurate regarding the services provided by local governments in the United States?"],
    ["The practice of hiring a foreign third-party service provider to run an operation is called"],
    ["Which one of the following items is an example of nonmaterial culture?"]
]

# Compute BLEU score
results = bleu.compute(predictions=predictions, references=references)

print(f"BLEU score: {results['score']:.2f}")
