from transformers import AutoTokenizer

# Load Qwen3.5-0.8B tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")

user_prompt = """You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.

Question: Given an integer n, output the sum of integers from 1 to n.

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.
python
# YOUR CODE HERE"""

messages = [
    {"role": "system", "content": "You are a helpful and precise assistant for solving programming problems."},
    {"role": "user", "content": user_prompt}
]

# print(tokenizer.chat_template)
print(tokenizer.special_tokens_map)

# Apply chat template
result = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

print(result)
