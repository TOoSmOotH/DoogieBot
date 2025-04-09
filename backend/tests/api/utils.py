import random
import string
from typing import Dict

def random_lower_string(length: int = 32) -> str:
    """Generate a random lowercase string."""
    return "".join(random.choices(string.ascii_lowercase, k=length))

def random_email() -> str:
    """Generate a random email address."""
    return f"{random_lower_string(length=10)}@{random_lower_string(length=8)}.com"

# Add any other common test utilities here if needed in the future.