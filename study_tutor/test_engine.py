from tutor_engine import explain_concept, socratic_chat

# Test 1
print("=== EXPLANATION MODE ===")
print(explain_concept("gradient descent"))

# Test 2
print("\n=== SOCRATIC MODE ===")
history = []
response, history = socratic_chat("Help me find the second largest element in an array", history)
print(f"Tutor: {response}")

response, history = socratic_chat("I would loop through the array", history)
print(f"Tutor: {response}")