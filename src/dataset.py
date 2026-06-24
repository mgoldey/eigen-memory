import random

def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_label(n):
    # Rule Priority:
    # 1. Prime -> RED
    # 2. Divisible by 5 -> BLUE
    # 3. Else -> GREEN
    if is_prime(n):
        return "RED"
    elif n % 5 == 0:
        return "BLUE"
    else:
        return "GREEN"

def generate_dataset(num_samples=100, seed=42):
    random.seed(seed)
    data = []
    # Generate a mix of numbers to ensure coverage of all classes
    # Range 1-100 to start simple
    for _ in range(num_samples):
        x = random.randint(1, 100)
        label = get_label(x)
        data.append({"input": x, "label": label})
    return data

if __name__ == "__main__":
    ds = generate_dataset(10)
    for item in ds:
        print(item)
