from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage
import requests
from bs4 import BeautifulSoup
from simpleeval import simple_eval


def respond_yourself(query: str) -> str:
    return query

def weather_str(location: str) -> str:
    # Dummy implementation, replace with actual weather fetching logic if needed
    return f"Current weather in {location} is sunny with a temperature of 75°F."

# Tools
def calculator_str(expr: str) -> str:
    try:
        return str(simple_eval(expr))
    except Exception:
        return "Error in calculation"

def web_search_str(query: str) -> str:
    url = f"https://html.duckduckgo.com/html/?q={query}"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(r.text, "html.parser")
    results = soup.find_all("a", class_="result__a", limit=3)
    return "\n".join([res.get_text() for res in results]) if results else "No results found."

# LLM
llm = ChatOllama(model="mistral:7b-instruct-q4_K_M", temperature=0, num_predict=200)

# Combined prompt: the LLM itself executes "tools"
def run_query(query: str) -> str:
    prompt = f"""Call the functions as needed, make sure to follow this 
    format exactly with () and ONLY CHOOSE 1 tool, the best tool for the job:

1. Calculator(query: str) -> returns the result of a math expression
2. WebSearch(query: str) -> returns top search results for a query
3. Weather(location: str) -> returns current weather information for a location
4. RespondYourself(query: str) -> returns a direct response without tools, use if stuck or confused

User query: {query}"""

    response = llm([HumanMessage(content=prompt)])
    text = response.content.strip()
    # print("\n\n" + text + "\n\n")

    # Post-process simulated tool calls
    # Replace simulated Calculator calls
    while "Calculator(" in text:
        expr = text.split("Calculator(")[1].split(")")[0]
        expr_orig = expr
        expr = expr.replace("\"", "")
        result = calculator_str(expr)
        text = result

    # Replace simulated WebSearch calls
    while "WebSearch(" in text:
        search_query = text.split("WebSearch(")[1].split(")")[0]
        search_query = search_query.replace("\"", "")
        result = web_search_str(search_query)
        text = result

    while "Weather(" in text:
        location = text.split("Weather(")[1].split(")")[0]
        location = location.replace("\"", "")
        result = weather_str(location)
        text = result

    while "RespondYourself(" in text:
        user_query = text.split("RespondYourself(")[1].split(")")[0]
        user_query = user_query.replace("\"", "")
        result = respond_yourself(user_query)
        text = result

    return text

questions = [
    "What is 25 * 4 + 12?",
    "Search the web for the latest Python version",
    "calculate 84 divided by 7 plus 3?",
    "what is a cat check the web?",
    "what is current weather report in glen cove today?",
    "chickens fly check the web?",
    "respond to me how are you today?",
    "I need you to tell me how are you doing today?",
    "chicken soup for days",
    "what is capital of France?"
]

for question in questions:
    print(f"Q: {question}")
    answer = run_query(question)
    print(f"A: {answer}\n")
