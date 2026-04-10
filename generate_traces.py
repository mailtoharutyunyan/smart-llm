#!/usr/bin/env python3
import random
import subprocess
import time

# A diverse set of starter questions to seed the trace generation
topics = [
    "quantum computing vs classical computing",
    "the philosophical implications of determinism",
    "how mRNA vaccines are designed",
    "trade-offs between monolith and microservice architectures",
    "the causes of the 2008 financial crisis",
    "how neural networks learn through backpropagation",
    "the impact of artificial intelligence on labor markets",
    "explain the theory of general relativity with analogies",
    "how distributed consensus protocols like Paxos work",
    "the historical context of the Renaissance",
    "mechanisms of natural selection and genetic drift",
    "the architecture of the James Webb Space Telescope",
    "compare and contrast different sorting algorithms",
    "the psychology of decision making under risk",
    "how renewable energy grids balance load",
]

print("Started Background Trace Generator...")
print("This will continually ask questions to build up the raw trace pool.")


def generate_question():
    topic = random.choice(topics)
    prompts = [
        f"Explain {topic} in deep technical detail.",
        f"What are the main criticisms of {topic}?",
        f"Provide a step-by-step breakdown of {topic}.",
        f"Compare the common misconceptions about {topic} with the reality.",
    ]
    return random.choice(prompts)


count = 0
try:
    while True:
        question = generate_question()
        print(f"\n[{count}] Generating trace for: {question}")

        # Run acs.py think to generate and auto-collect the trace
        subprocess.run(["venv/bin/python", "acs.py", "think", question])

        count += 1
        time.sleep(2)  # brief pause to prevent overheating/rate limits

except KeyboardInterrupt:
    print(f"\nStopped. Generated {count} trace requests.")
