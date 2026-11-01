import argparse
from dataclasses import dataclass
import inspect
import os
import sys
from json import loads
from subprocess import run, PIPE, STDOUT, TimeoutExpired
import shlex
from json import load
from pathlib import Path

import frontmatter
from openai import OpenAI
from rich import print


# Costants
BASE_DIR = Path(__file__).resolve().parent
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")
TOOLS_PATH  = BASE_DIR / "tools.json"
SKILLS_PATH = Path(".claude/skills")

# LLM Model to use
MODEL = "anthropic/claude-haiku-4.5"

@dataclass
class Skill:
	""" Skill data """
	name: str
	desc: str
	body: str

	def __str__(self) -> str:
		return f"- {self.name}: {self.desc}"


def load_tools() -> list[dict[str, str]]:
    """ loads tools from disk """
    tools = []
    try:
        with open(TOOLS_PATH, mode='r', encoding="utf-8") as fp:
            tools = load(fp)
    except FileNotFoundError:
        print(err)
    return tools


def load_skills() -> list[Skill]:
    """ loads skills from disk """
    skills = []
    try:
        for skill_file in SKILLS_PATH.iterdir():
            try:
                with open(skill_file / "SKILL.md") as fd:
                    src = frontmatter.load(fd)
                    skills.append(
                        Skill(
                            src.metadata["name"],
                            src.metadata["description"],
                            ""
                        )
                    )
            except FileNotFoundError:
                print(f"skill not set for {skill_file.name}")
    except FileNotFoundError:
        #print("no skills found")
        pass
    return skills


def invoke_skill(skill: str) -> str|None:
    try:
        full_skill_path = (SKILLS_PATH / skill / "SKILL.md").resolve()
        #if not str(full_skill_path).startswith(str(SKILLS_PATH)):
        #    raise ValueError("Directory traversal attempt")
        with open(full_skill_path) as fd:
            src = frontmatter.load(fd)
            return src.content
    except FileNotFoundError:
        print(f"provided '{skill}' was not found")

############################### TOOL'S DEFINITION ###################################

def tool_file_read(params: dict[str, str]) -> str|None:
    """read the provided file from the local system"""
    file = params.get("file_path", None)

    if not file:
        raise RuntimeError("file_path is required")

    with open(file, mode="r", encoding="utf-8") as fd:
        return fd.read()


def tool_file_write(params: dict[str, str]) -> str|None:
    """ write the provided file on the local system """
    file, content = params.get("file_path", None), params.get("content", None)
    if not file or not content:
        raise RuntimeError("file and content are required")
    with open(file, mode='w', encoding='utf-8') as fd:
        fd.write(content)
        return f"{file} written with success"


def tool_command_execute(params: dict[str, str]) -> str|None:
    """ execute a system command """
    command = params.get("command", None)
    if not command:
        raise RuntimeError("command not provided")
    try:
        proc = run(
            shlex.split(command),
            text=True,
            stdout=PIPE,
            stderr=STDOUT,
            timeout=0.5
        )
        return proc.stdout
    except TimeoutExpired:
        raise RuntimeError("command timeout")

############################### END TOOL'S DEFINITION ###############################

def main():
    current_functions = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    available_tools = {name: func for name, func in current_functions if name.startswith("tool_")}
    
    tools = load_tools()
    skills = load_skills()
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    messages = [
        {
            "role": "system",
            "content": f"You have access to the following skills:\n\n{'\n'.join(map(str, skills))}"
        },
    ]

    p = argparse.ArgumentParser()
    p.add_argument(
        "-p",
        "--prompt",
        required=True
    )
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    if args.prompt.startswith("/"):
        skill_desc = invoke_skill(args.prompt[1:])
        messages.append(
            {
                "role": "user",
                "content": skill_desc
            }
        )
    else:
        messages.append(
            {
                "role": "user",
                "content": args.prompt
            }
        )

    response = None
    try:
        while True:
            response = client.responses.create(
                model=MODEL,
                input=messages,
                tools=tools
            )

            messages += response.output
            function_calls = [
                item for item in response.output
                if item.type == "function_call"
            ]

            # if no more functions call terminate
            if not function_calls:
                print(response.output_text, file=sys.stdout)
                break

            # execute tools
            for item in function_calls:
                tool = available_tools.get(item.name)

                if not tool:
                    raise RuntimeError(
                        f"Requested tool '{item.name}' not registered"
                    )

                try:
                    result = tool(loads(item.arguments))
                except Exception as exc:
                    result = f"Error executing tool: {exc}"

                messages.append({
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": str(result)
                })
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
