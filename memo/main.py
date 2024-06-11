import sys
import os

from .memorization_check import main
from .memorization_check_2 import main as main2

if __name__ == "__main__":
    if len(sys.argv) > 1:
        second_value = sys.argv[1]

        match second_value:
            case 'tomi':
                filename = os.path.join("data", "tomi", "test.txt")
                main(filename)
            case 'fantom':
                filename = os.path.join("data", "fantom", "fantom_v1.json")
                main(filename)
            case 'adv-csfb':
                filename = os.path.join("data", "Adv-CSFB", "unexpected_contents.jsonl")
                main(filename)
            case 'socialiqa':
                filename = os.path.join("data", "socialIQa", "dev.jsonl")
                main(filename)
            case 'mindgames':
                filename = os.path.join("data", "mindgames", "test-00000-of-00001-7dfe9e22268ffc8b.csv")
                main2(filename)
            case _:
                print("Second value is not in the desired values.")
