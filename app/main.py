import argparse
import inspect
import os
import sys
from json import loads
from subprocess import run, PIPE, STDOUT, TimeoutExpired
import shlex

from openai import OpenAI
from rich import print

# Costants
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

######################################### TOOL'S DEFINITION ###########################
def tool_read(params: dict[str, str]) -> str|None:
    """read the provided file from the local system"""
    file = params.get("file_path", None)

    if not file:
        raise RuntimeError("file_path is required")

    with open(file, mode="r", encoding="utf-8") as fd:
        return fd.read()

def tool_write(params: dict[str, str]) -> str|None:
    """ write the provided file on the local system """
    file, content = params.get("file_path", None), params.get("content", None)
    if not file or not content:
        raise RuntimeError("file and content are required")
    with open(file, mode='w', encoding='utf-8') as fd:
        fd.write(content)
        return f"{file} written with success"

def tool_bash(params: dict[str, str]) -> str|None:
    """ execute a system command """
    command = params.get("command", None)
    if not command:
        raise RuntimeError("command not provided")
    try:
        proc = run(
            shlex.split(command),
            universal_newlines=True,
            stdout=PIPE,
            stderr=STDOUT,
            timeout=0.5
        )
        return proc.stdout
    except TimeoutExpired:
        raise RuntimeError("command timeout")

######################################### END TOOL'S DEFINITION #######################
def main():
    current_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    tools = {name: func for name, func in current_functions if name.startswith("tool_")}

    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": args.p}]

    while True:
        response = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
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
                                    "description": "The path to the file to read",
                                }
                            },
                            "required": ["file_path"],
                        },
                    },
                },
                {
                  "type": "function",
                  "function": {
                    "name": "write",
                    "description": "Write content to a file",
                    "parameters": {
                      "type": "object",
                      "required": ["file_path", "content"],
                      "properties": {
                        "file_path": {
                          "type": "string",
                          "description": "The path of the file to write to"
                        },
                        "content": {
                          "type": "string",
                          "description": "The content to write to the file"
                        }
                      }
                    }
                  }
                },
                {
                  "type": "function",
                  "function": {
                    "name": "bash",
                    "description": "Execute a shell command",
                    "parameters": {
                      "type": "object",
                      "required": ["command"],
                      "properties": {
                        "command": {
                          "type": "string",
                          "description": "The command to execute"
                        }
                      }
                    }
                  }
                }
            ],
        )

        actor = response.choices[0].message
        messages.append(actor)

        #if not res.choices or len(res.choices) == 0:
        #    raise RuntimeError("no choices in response")
        if actor.tool_calls:
            # if the model want to use any tool search for available tools
            # and invoke it
            for tool in actor.tool_calls:
                if invoke := tools.get(f"tool_{tool.function.name}", None):
                    result = invoke(loads(tool.function.arguments))
                    messages.append(
                        {"role": "tool", "tool_call_id": tool.id, "content": result}
                    )
                else:
                    raise RuntimeError(f"Requested tool {tool.function.name} not registered")
            continue
        print(actor.content, file=sys.stdout)
        break


if __name__ == "__main__":
    main()
