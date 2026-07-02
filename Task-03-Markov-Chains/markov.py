# markov.py
# Task-03: Text Generation with Markov Chains
# Builds a statistical model that predicts the next word based on previous word(s)

import random
import re
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
ORDER       = 2        # how many previous words to look at (1=simple, 2=better quality)
OUTPUT_LEN  = 100      # number of words to generate
SEED        = 42       # for reproducibility

# ─────────────────────────────────────────────
# SAMPLE TRAINING TEXT (Shakespeare style)
# ─────────────────────────────────────────────
TRAINING_TEXT = """
To be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles
and by opposing end them to die to sleep no more and by a sleep to say we end
the heartache and the thousand natural shocks that flesh is heir to tis a consummation
devoutly to be wished to die to sleep to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come when we have shuffled off this mortal coil
must give us pause there is the respect that makes calamity of so long life
for who would bear the whips and scorns of time the oppressors wrong the proud mans contumely
the pangs of despised love the laws delay the insolence of office and the spurns
that patient merit of the unworthy takes when he himself might his quietus make
with a bare bodkin who would fardels bear to grunt and sweat under a weary life
but that the dread of something after death the undiscovered country from whose bourn
no traveller returns puzzles the will and makes us rather bear those ills we have
than fly to others that we know not of thus conscience does make cowards of us all
and thus the native hue of resolution is sicklied over with the pale cast of thought
and enterprises of great pitch and moment with this regard their currents turn awry
and lose the name of action
To be or not to be a noble thought to dream of life and death and love and fate
the king is dead long live the king the queen speaks softly in the hall
what light through yonder window breaks it is the east and juliet is the sun
arise fair sun and kill the envious moon who is already sick and pale with grief
that thou her maid art far more fair than she be not her maid since she is envious
her vestal livery is but sick and green and none but fools do wear it cast it off
it is my lady it is my love oh that she knew she were she speaks yet she says nothing
what of that her eye discourses i will answer it i am too bold tis not to me she speaks
two of the fairest stars in all the heaven having some business do entreat her eyes
to twinkle in their spheres till they return what if her eyes were there they in her head
the brightness of her cheek would shame those stars as daylight doth a lamp her eyes in heaven
would through the airy region stream so bright that birds would sing and think it were not night
see how she leans her cheek upon her hand oh that i were a glove upon that hand
that i might touch that cheek she speaks oh speak again bright angel for thou art
as glorious to this night being over my head as is a winged messenger of heaven
unto the white upturned wondering eyes of mortals that fall back to gaze on him
when he bestrides the lazy puffing clouds and sails upon the bosom of the air
"""

# ─────────────────────────────────────────────
# MARKOV CHAIN BUILDER
# ─────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()

def build_markov_chain(words, order=2):
    chain = defaultdict(list)
    for i in range(len(words) - order):
        key   = tuple(words[i:i + order])
        value = words[i + order]
        chain[key].append(value)
    return chain

def generate_text(chain, order=2, length=100, seed=None):
    if seed:
        random.seed(seed)

    # pick a random starting key
    start = random.choice(list(chain.keys()))
    result = list(start)

    for _ in range(length - order):
        key = tuple(result[-order:])
        if key not in chain:
         start = random.choice(list(chain.keys()))
        result.extend(list(start)) 
        continue
        next_word = random.choice(chain[key])
        result.append(next_word)

    return ' '.join(result)

# ─────────────────────────────────────────────
# TRAIN & GENERATE
# ─────────────────────────────────────────────
print("=" * 55)
print("Task-03: Text Generation with Markov Chains")
print("=" * 55)

words = clean_text(TRAINING_TEXT)
print(f"Training text: {len(words)} words")

chain = build_markov_chain(words, order=ORDER)
print(f"Markov chain built: {len(chain)} unique states (order={ORDER})")

print("\n--- Generated Text (Order 1) ---")
chain1 = build_markov_chain(words, order=1)
print(generate_text(chain1, order=1, length=OUTPUT_LEN, seed=SEED))

print("\n--- Generated Text (Order 2) ---")
chain2 = build_markov_chain(words, order=2)
print(generate_text(chain2, order=2, length=OUTPUT_LEN, seed=SEED))

print("\n--- Generated Text (Order 3) ---")
chain3 = build_markov_chain(words, order=3)
print(generate_text(chain3, order=3, length=OUTPUT_LEN, seed=SEED))

# ─────────────────────────────────────────────
# INTERACTIVE MODE
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("Interactive mode — generate your own text")
print("=" * 55)

while True:
    try:
        length = input("\nHow many words to generate? (or 'quit'): ").strip()
        if length.lower() in ('quit', 'q', 'exit'):
            break
        length = int(length) if length.isdigit() else 100
        order  = input("Order? (1=random, 2=better, 3=best) [2]: ").strip()
        order  = int(order) if order in ('1','2','3') else 2

        chain_i = build_markov_chain(words, order=order)
        print("\n" + generate_text(chain_i, order=order, length=length))
    except KeyboardInterrupt:
        break

print("\nDone!")
