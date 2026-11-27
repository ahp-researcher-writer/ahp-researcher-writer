from dotenv import load_dotenv
load_dotenv()
import os
from langchain_core.messages import HumanMessage
from research_agent import agent

query = (
    "Provide a factual overview of the main branches of Islam (Sunni, Shia, Ibadi). "
    "For each branch, cover: historical origins and the key events behind the split, "
    "core theological and legal differences (including major schools of jurisprudence), "
    "approximate global population share, and main geographic distribution. "
    "Use scholarly or authoritative sources (academic publications, Pew Research, "
    "encyclopedias) and present the branches neutrally without favoring any."
)

result = agent.invoke({"messages": [HumanMessage(content=query)]})

report = result.get("final_report")
if not report:  # the agent asked a clarifying question instead of researching
    print("Clarification needed:\n", result["messages"][-1].content)
else:
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/research_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    import json
    with open("outputs/research_brief.json", "w", encoding="utf-8") as f:
        json.dump(result["research_output"], f, indent=2, ensure_ascii=False)
    print(report)
    print("Saved report to outputs/research_report.md")
