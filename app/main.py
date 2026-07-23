import argparse
import os
import sys
import inspect
from json import loads
from openai import OpenAI

# Costants
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

######################################### TOOL'S DEFINITION ###########################
def tool_read(params: Dict[str, str]) -> None:
    """ read the provided file from the local system """
    if file := params.get("file_path", None):
        with open(file, mode='r', encoding="utf-8") as fd:
            print(fd.read(), file=sys.stdout)

######################################### END TOOL'S DEFINITION #######################
def main():
    current_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    tools = {name:func for name, func in current_functions if name.startswith("tool_")}
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        # TODO: Move into specifics configuration file
        tools=[
            {
              "type": "function",
              "function": {
                "name": "read",
                "description": "Read and return the contents of a file",
                "parameters": {
                  "type": "object",
                  "properties": {
                    "file_path": {
                      "type": "string",
                      "description": "The path to the file to read"
                    }
                  },
                  "required": ["file_path"]
                }
              }
            }
        ]
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    
    if msg := chat.choices[0].message.content:
        print(msg)
    else:
        # is tool invokation?
        for tool in chat.choices[0].message.tool_calls:
            if invoke := tools.get(f"tool_{tool.function.name}", None):
                invoke(loads(tool.function.arguments))


if __name__ == "__main__":
    main()
