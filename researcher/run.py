from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from langchain_core.messages import HumanMessage
from research_agent import researcher_agent


research_brief = "I want to identify and evaluate the coffee shops in San Francisco that are considered the best based specifically on coffee quality."

result = researcher_agent.invoke(
    {"researcher_messages": [HumanMessage(content=f"{research_brief}.")]}
)

print(result["compressed_research"])

with open("output/research_report.md", "w", encoding="utf-8") as f:
    f.write(result["compressed_research"])

print("Saved report to research_report.md")