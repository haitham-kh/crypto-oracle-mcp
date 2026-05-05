"""Ad-hoc runner: invoke oracle_intelligence_brief for a given query."""
import asyncio, json, sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from tools_advanced import tool_oracle_intelligence_brief

QUERY = sys.argv[1] if len(sys.argv) > 1 else "OG"
TAG = sys.argv[2] if len(sys.argv) > 2 else QUERY

async def main():
    try:
        res = await tool_oracle_intelligence_brief(query=QUERY, format="both")
    except Exception as e:
        traceback.print_exc()
        res = {"error": str(e)}
    out_path = os.path.join(os.path.dirname(__file__), f"_brief_{TAG}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, default=str, indent=2)
    md = None
    if isinstance(res, dict):
        md = (res.get("data") or {}).get("brief_markdown") or res.get("markdown")
    if md:
        with open(os.path.join(os.path.dirname(__file__), f"_brief_{TAG}.md"), "w", encoding="utf-8") as f:
            f.write(md)
        print("---MD---")
        print(md)
    else:
        print("NO MARKDOWN. Payload keys:", list(res.keys()) if isinstance(res, dict) else type(res))
    print("WROTE", out_path)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
