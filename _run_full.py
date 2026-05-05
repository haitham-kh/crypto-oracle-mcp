"""Ad-hoc runner: invoke full_coin_intelligence_report for a given query."""
import asyncio, json, sys, os, traceback
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

from tools_advanced import tool_full_coin_intelligence_report

QUERY = sys.argv[1] if len(sys.argv) > 1 else "OG"
TAG = sys.argv[2] if len(sys.argv) > 2 else QUERY

async def main():
    try:
        res = await tool_full_coin_intelligence_report(query=QUERY)
    except Exception as e:
        traceback.print_exc()
        res = {"error": str(e)}
    out_path = os.path.join(os.path.dirname(__file__), f"_full_{TAG}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, default=str, indent=2)
    print("WROTE", out_path)
    if isinstance(res, dict):
        print("success:", res.get("success"))
        print("top-level keys:", list(res.keys()))
        data = res.get("data") or {}
        if isinstance(data, dict):
            print("data keys:", list(data.keys()))
            failed = res.get("sources_failed") or data.get("failed_tools") or []
            if failed:
                print("failed tools:", failed)

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
