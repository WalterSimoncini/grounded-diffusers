import random

from typing import Dict, List


def visual_adjectives_prompt(
    label: str,
    visual_adjectives: Dict[str, Dict[str, List[str]]],
    max_adjectives=2
):
    picked_adjectives = []
    class_adjectives = visual_adjectives[label]

    for adjective_group in class_adjectives.keys():
        picked_adjectives.append(
            random.choice(class_adjectives[adjective_group])
        )

    # Pick at most max_adjectives
    picked_adjectives = random.sample(
        picked_adjectives,
        k=min(max_adjectives, len(picked_adjectives))
    )

    picked_adjectives = " ".join(picked_adjectives)

    return f"{picked_adjectives} {label}"
