import spacy

# Load English NLP model from spaCy
#python -m spacy download en_core_web_sm     from Anaconda prompt
nlp = spacy.load("en_core_web_sm")

# Sample text data (approx. 100 words) about an automobile company
text = """
Tesla Inc., founded by Elon Musk in 2003, is a pioneering electric vehicle and clean energy company headquartered in Palo Alto, California. 
Tesla's flagship products include the Model S, Model 3, Model X, and Model Y electric cars, as well as battery energy storage systems 
like the Powerwall, Powerpack, and Megapack. The company also operates Gigafactories in Nevada, Shanghai, and Berlin to scale its 
manufacturing capabilities. In 2020, Tesla joined the S&P 500 index and has since become one of the world's most valuable companies. 
With advancements in autonomous driving, Tesla's Full Self-Driving (FSD) software is seen as the future of mobility.
"""

# Process the text
doc = nlp(text)

# Extract named entities
entities = [(ent.text, ent.label_) for ent in doc.ents]

entities

nlp = spacy.load("en_core_web_sm")
print(nlp.get_pipe("ner").labels)


##################################
#Descriptions
# "CARDINAL": "Numerals that do not fall under another type.",
#     "DATE": "Absolute or relative dates or periods.",
#     "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
#     "FAC": "Buildings, airports, highways, bridges, etc.",
#     "GPE": "Countries, cities, states.",
#     "LANGUAGE": "Named languages.",
#     "LAW": "Named laws or legal documents.",
#     "LOC": "Non-GPE locations, mountain ranges, bodies of water.",
#     "MONEY": "Monetary values, including units.",
#     "NORP": "Nationalities, religious or political groups.",
#     "ORDINAL": "‘first’, ‘second’, etc.",
#     "ORG": "Companies, agencies, institutions, etc.",
#     "PERCENT": "Percentage values.",
#     "PERSON": "People, including fictional characters.",
#     "PRODUCT": "Objects, vehicles, foods, etc. (Not services.)",
#     "QUANTITY": "Measurements such as weight, distance, etc.",
#     "TIME": "Times smaller than a day.",
#     "WORK_OF_ART": "Titles of books, songs, artworks, etc."