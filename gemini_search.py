import os
from google.colab import userdata
from google import genai
from IPython.display import HTML, Markdown

# Setting the API Key from Google Gemini:
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# Setting the SDK client from Gemini:
client = genai.Client()

MODEL_ID = "gemini-2.0-flash"

# Ask a recent question:
response = client.models.generate_content(
    model=MODEL_ID,
    contents="When it's the next NBA game?"
)

# Print the answer:
display(Markdown(f"Answer:\n {response.text}"))

# Asking Gemini the same question, now using Google search:
response = client.models.generate_content(
    model=MODEL_ID,
    contents="When it's the next NBA game?",
    config={"tools": [{"google_search": {}}]}
)

# Print the answer:
display(Markdown(f"Answer using Google Search:\n {response.text}"))

# Print the search:
print(f"Search executed: {response.candidates[0].grounding_metadata.web_search_queries}")
# Print the 'URLs' used in the search:
print(f"Webpage(s) used in the search: {', '.join([site.web.title for site in response.candidates[0].grounding_metadata.grounding_chunks])}")
print()
display(HTML(response.candidates[0].grounding_metadata.search_entry_point.rendered_content))
