"""AskMoMA: a conversational guide to the art on view at the Museum of Modern Art.

Group 4, MSAI-631-M80, University of the Cumberlands, Fall 2026.

Layout:
    config.py         paths and constants
    data/             MoMA's open data: fetch.py downloads it, collection.py loads it
    features/         one module per use case (locate.py is the first)
    chat/router.py    decides which feature answers a message
    ui/app.py         the Gradio chat page
"""

__version__ = "0.1.0"
