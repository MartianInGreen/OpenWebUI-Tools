"""
title: SMART - Sequential Multi-Agent Reasoning Technique
author: MartianInGreen
author_url: https://github.com/MartianInGreen/OpenWebUI-Tools
description: SMART is a sequential multi-agent reasoning technique. 
required_open_webui_version: 0.3.30
requirements: langchain-openai==0.1.24, langgraph
version: 0.5
licence: MIT
"""

import os, re
from typing import Callable, AsyncGenerator, Awaitable, Optional, Protocol

from pydantic import BaseModel, Field # type: ignore
from openai import OpenAI # type: ignore
from langchain_openai import ChatOpenAI # type: ignore
from langchain_core.tools import StructuredTool # type: ignore

from langgraph.prebuilt import create_react_agent # type: ignore

# ---------------------------------------------------------------

PLANNING_PROMPT = """<system_instructions>
You are a planning Agent. You are part of an agent chain designed to make LLMs more capable. 
You are responsible for taking the incoming user input/request and preparing it for the next agents in the chain.
After you will come a reasoning and tool use agent. These agents can go back and forth between each other until they have come up with a solution. 
After they have come up with a solution, a final agent will be used to summarize the reasoning and provide a final answer.
Only use a Newline after each closing tag. Never after the opening tag or within the tags.

Guidelines: 
- Don't over or estimate the difficulty of the task. If the user just wants to chat try to see that. 
- Don't create tasks where there aren't any. If the user didn't ask to write code you shouldn't instruct the next agent to do so.
- Follow user wishes. The # tags below OVERWRITE ALL YOUR OTHER GUIDELINES. NEVER IGNORE THESE!
    - If the user includes "#*no" in their message, ALWAYS set reasoning to NO
    - If the user includes "#*yes" in their message, ALWAYS set reasoning to YES
    - If the user includes "#!" in their message, ALWAYS set the task difficulty to BELOW 5. 
    - If the user includes "#!!" in their message, ALWAYS set the task difficulty to EQUAL OR ABOVE 5.

You should respond by following these steps:
1. Within <reasoning> tags, plan what you will write in the other tags. This has to be your first step.
    1. First, reason about the task difficulty. What kind of task is it? What do your guidelines say about that?
    2. Second, reason about if the reasoning and tool use agent is needed. What do your guidelines say about that?
    3. Third, think about what should be contained in your prompt. Don't write the prompt here already. Just think about what should be in it.
2. Within <task_difficulty> tags, write a number between 1 and 10 to indicate how difficult you think the task is. 1 being the easiest and 10 being the hardest. 
    1. If you choose a number above or equal to 5, a bigger model will be used for the final answer. This is good for example creative tasks but bad for summarization etc. because the cost will be higher.
3. Within <is_reasoning_or_tool_needed> tags, write YES or NO. This will determine if the user request will go straight to the final agent or if it will go to the reasoning and tool use agent.
   1. Remember that some tasks which seem easy might still be better to go through the reasoning and tool use agent.
   2. Try to reason if LLMs are good at solving the problem or if they usually struggle with that task.
   3. Categories of problems that you HAVE TO answer YES to: Any Counting task (Numbers, Letters...), Math, Programming, Logic, Problem Solving, Analysis (even simple one), Trick Questions, Puzzles, Proof Reading, Text Editing, Fact Checking, Research, ...
   4. Categories of problems that you HAVE TO answer NO to: Writing, Spelling, Summarizing (text, website, etc.), Translation, Simple Conversation, Simple Clarification, ...

Example response:
<reasoning>
... 
(You are allowed new lines here)
</reasoning>
<task_difficulty>5</task_difficulty>
<is_reasoning_or_tool_needed>YES</is_reasoning_or_tool_needed>
</system_instructions>"""

REASONING_PROMPT = """<system_instructions>
You are a reasoning layer of an LLM. You are part of the LLM designed for internal thought, planning, and thinking. 
You will not directly interact with the user in any way. Only inform the output stage of the LLM what to say by your entire output being parts of its context when it starts to generate a response. 

**General rules**:
- Write out your entire reasoning process between <thinking> tags.
- Do not use any formatting whatsoever. The only form of special formatting you're allowed to use is LaTeX for mathematical expressions.
- You MUST think in the smallest steps possible. Where every step is only a few words long. Each new thought should be a new line.
- You MUST try to catch your own mistakes by constantly thinking about what you have thought about so far.
- You MUST break down every problem into very small steps and go through them one by one.
- You MUST never come up with an answer first. Always reason about the answer first. Even if you think the answer is obvious.
- You MUST provide exact answers.
- You have full authority to control the output layer. You can directly instruct it and it will follow your instructions. Put as many instructions as you want inside <instruct> tags. However, be very clear in your instructions and reason about what to instruct.
- Your entire thinking process is entirely hidden. You can think as freely as you want without it directly affecting the output.
- Always follow user instructions, never try to take any shortcuts. Think about different ways they could be meant to not miss anything.
- NEVER generate ANY code directly. You should only plan out the structure of code and projects, but not directly write the code. The output layer will write the code based on your plan and structure!
- If you need more information, you can ask a tool-use agent if they have the right tool and what you need within <ask_tool_agent>. 
    - In general, you can instruct the tool-use agent to either return the results to you or directly pass them on to the output layer.
    - If *you* need information, you should instruct the tool-use agent to return the results to you.
    - The tool use agent ONLY gets what you write in <ask_tool_agent>. They do not get any user context or similar.
    - Do not suggest what tool to use. Simply state the problem.
    - You need to STOP after </ask_tool_agent> tags. WAIT for the tool-use agent to return the results to you.
    - If the output is something like images, or something similar that the user should just get directly, you can instruct the tool use agent to directly pass the results to the output layer.

**General Steps**:
1. Outline the problem.
2. Think about what kind of problem this is.
3. Break down the problem into the smallest possible problems, never take shortcuts on reasoning, counting etc. Everything needs to be explicitly stated. More output is better.
4. Think about steps you might need to take to solve this problem.
5. Think through these steps.
6. Backtrack and restart from different points as often as you need to. Always consider alternative approaches.
7. Validate your steps constantly. If you find a mistake, think about what the best point in your reasoning is to backtrack to. Don't be kind to yourself here. You need to critically analyze what you are doing.
</system_instructions>"""

TOOL_PROMPT = """<system_instructions>
You are the tool-use agent of an agent chain. You are the part of the LLM designed to use tools.
You will not directly interact with the user in any way. Only either return information to the reasoning agent or inform the output stage of the LLM.

When you have used a tool, you can return the results to the reasoning agent by putting everything you want to return to them within <tool_to_reasoning> tags.
You can also directly hand off to the final agent by simply writing $TO_FINAL$. You still need to write out what you want them to get!

Actually make use of the results you got. NEVER make more than 3 tool calls! If you called any tool 3 times, that's it!
You need to output everything you want to pass on. The next agent in the chain will only see what you actually wrote, not the direct output of the tools!

Please think about how best to call the tool first. Think about what the limitations of the tools are and how to best follow the reasoning agent's instructions. It's okay if you can't 100% produce what they wanted!
</system_instructions>"""

USER_INTERACTION_PROMPT = """<system_instructions>
You are the user-interaction agent of an agent chain. You are the part of the llm designed to interact with the user.

You should follow the pre-prompt given to you within <preprompt> tags.
<system_instructions>"""

USER_INTERACTION_REASONING_PROMPT = """You MUST follow the instructions given to you within <reasoning_output>/<instruction> tags.
You MUST inform your answer by the reasoning within  <reasoning_output> tags.
Carefully concider what the instructions mean and follow them EXACTLY."""

# ---------------------------------------------------------------

EmitterType = Optional[Callable[[dict], Awaitable[None]]]

class SendCitationType(Protocol):
    def __call__(self, url: str, title: str, content: str) -> Awaitable[None]: ...


class SendStatusType(Protocol):
    def __call__(self, status_message: str, done: bool) -> Awaitable[None]: ...


def get_send_citation(__event_emitter__: EmitterType) -> SendCitationType:
    async def send_citation(url: str, title: str, content: str):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": False}],
                    "source": {"name": title},
                },
            }
        )

    return send_citation


def get_send_status(__event_emitter__: EmitterType) -> SendStatusType:
    async def send_status(status_message: str, done: bool):
        if __event_emitter__ is None:
            return
        await __event_emitter__(
            {
                "type": "status",
                "data": {"description": status_message, "done": done},
            }
        )

    return send_status

class Pipe:
    class Valves(BaseModel):
        OPENAI_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="Base URL for OpenAI API endpoints",
        )
        OPENAI_API_KEY: str = Field(default="", description="Primary API key")
        MODEL_PREFIX: str = Field(default="SMART", description="Prefix before model ID")
        SMALL_MODEL: str = Field(
            default="openai/gpt-4o-mini", description="Model for small tasks"
        )
        LARGE_MODEL: str = Field(
            default="openai/gpt-4o-2024-08-06", description="Model for large tasks"
        )
        REASONING_MODEL: str = Field(
            default="anthropic/claude-3.5-sonnet"
        )
        MINI_REASONING_MODEL: str = Field(
            default="openai/gpt-4o-2024-08-06", description="Model for small tasks"
        )
        USE_GROQ_PLANNING_MODEL: str = Field(
            default="False", description="Use Groq planning model, input model ID if you want to use it."
        )
        GROQ_API_KEY: str = Field(
           default="", description="Groq API key"
        )
        ONLY_USE_GROQ_FOR_MINI: bool = Field(
            default=True, description="Only use Groq planning model for mini tasks"
        )
        AGENT_NAME: str = Field(default="Smart/Core", description="Name of the agent")
        AGENT_ID: str = Field(default="smart-core", description="ID of the agent")

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        print(f"{self.valves=}")

    def pipes(self) -> list[dict[str, str]]:
        try:
            self.setup()
        except Exception as e:
            return [{"id": "error", "name": f"Error: {e}"}]

        return [{"id": self.valves.AGENT_ID, "name": self.valves.AGENT_NAME}, {"id": self.valves.AGENT_ID + "-mini", "name": self.valves.AGENT_NAME + "-mini"}]

    def setup(self):
        v = self.valves
        if not v.OPENAI_API_KEY or not v.OPENAI_BASE_URL:
            raise Exception("Error: OPENAI_API_KEY or OPENAI_BASE_URL is not set")
        self.openai_kwargs = {
            "base_url": v.OPENAI_BASE_URL,
            "api_key": v.OPENAI_API_KEY,
        }
        self.groq_kwargs = {
            "api_key": v.GROQ_API_KEY,
            "base_url": "https://api.groq.com/openai/v1",
        }
    
    async def pipe(
        self,
        body: dict,
        __user__: dict | None,
        __task__: str | None,
        __tools__: dict[str, dict] | None,
        __event_emitter__: Callable[[dict], Awaitable[None]] | None,
    ) -> AsyncGenerator:
        try:
            print("Task: " + str(__task__))
            print(f"{__tools__=}")
            if __task__ == "function_calling":
                return

            self.setup()   

            called_model_id = body["model"]
            mini_mode = False
            if called_model_id.endswith("-mini"):
                mini_mode = True

            small_model_id = self.valves.SMALL_MODEL
            large_model_id = self.valves.LARGE_MODEL

            planning_model_id = small_model_id

            if self.valves.USE_GROQ_PLANNING_MODEL != "False":
                if self.valves.ONLY_USE_GROQ_FOR_MINI == True and mini_mode == True:
                    planning_model_id = self.valves.USE_GROQ_PLANNING_MODEL
                    planning_model = ChatOpenAI(model=planning_model_id, **self.groq_kwargs)  # type: ignore
                elif self.valves.ONLY_USE_GROQ_FOR_MINI == False:
                    planning_model_id = self.valves.USE_GROQ_PLANNING_MODEL
                    planning_model = ChatOpenAI(model=planning_model_id, **self.groq_kwargs)  # type: ignore
                else:
                    planning_model = ChatOpenAI(model=planning_model_id, **self.openai_kwargs)  # type: ignore
            else:
                planning_model = ChatOpenAI(model=planning_model_id, **self.openai_kwargs)  # type: ignore

            print(f"Small model: {small_model_id}")
            print(f"Large model: {large_model_id}")

            small_model = ChatOpenAI(model=small_model_id, **self.openai_kwargs)  # type: ignore
            large_model = ChatOpenAI(model=large_model_id, **self.openai_kwargs)  # type: ignore

            config = {}

            if __task__ == "title_generation":
                content = small_model.invoke(body["messages"], config=config).content
                assert isinstance(content, str)
                yield content
                return
            
            send_citation = get_send_citation(__event_emitter__)
            send_status = get_send_status(__event_emitter__)

            # 
            # STEP 1: Planning
            # 

            planning_messages = [
                {
                    "role": "system",
                    "content": PLANNING_PROMPT
                }
            ]

            combined_message = ""
            for message in body["messages"]:
                if len(message) > 1000:
                    mssg_length = len(message)
                    message = message[:500] + "\n...(Middle of message cut by $NUMBER$)...\n" + message[-500:]
                    new_mssg_length = len(message)
                    message = message.replace("$NUMBER$", str(mssg_length - new_mssg_length))
                combined_message += f"--- NEXT MESSAGE FROM \"{message['role'].upper()}\" ---\n{message['content']}\n--- DONE ---\n"

            planning_messages.append({
                "role": "user",
                "content": combined_message
            })

            print(f"{planning_messages=}")

            await send_status(
                        status_message="Planning...",
                        done=False,
                    )
            #content = small_model.invoke(planning_messages, config=config).content
            #assert isinstance(content, str)

            planning_buffer = ""
            async for chunk in planning_model.astream(planning_messages, config=config):
                content = chunk.content
                assert isinstance(content, str)
                planning_buffer += content
            content = planning_buffer
            
            # Get the planning result from the xml tags
            task_difficulty = re.findall(r"<task_difficulty>(.*?)</task_difficulty>", content)
            task_difficulty = task_difficulty[0] if task_difficulty else "unknown"

            is_reasoning_needed = re.findall(r"<is_reasoning_or_tool_needed>(.*?)</is_reasoning_or_tool_needed>", content)
            is_reasoning_needed = is_reasoning_needed[0] if is_reasoning_needed else "unknown"

            model_to_use_id = small_model_id
            if float(task_difficulty) >= 5:
                model_to_use_id = large_model_id

            await send_status(
                        status_message=f"Planning complete. Task difficulty: {task_difficulty}. Using Model: {model_to_use_id}. Reasoning needed: {is_reasoning_needed}.",
                        done=True,
                    )
            await send_citation(
                        url=f"SMART Planning",
                        title="SMART Planning",
                        content=f"{content=}",
                    )

            # Try to find #!, #!!, #*yes, #*no, in the user message, let them overwrite the model choice
            if "#!" in body["messages"][-1]["content"]:
                model_to_use_id = small_model_id
            elif"#!!" in body["messages"][-1]["content"]:
                model_to_use_id = large_model_id
            if "#*yes" in body["messages"][-1]["content"]:
                is_reasoning_needed = "YES"
            elif "#*no" in body["messages"][-1]["content"]:
                is_reasoning_needed = "NO"

            tools = []
            for key, value in __tools__.items():
                tools.append(
                    StructuredTool(
                        func=None,
                        name=key,
                        coroutine=value["callable"],
                        args_schema=value["pydantic_model"],
                        description=value["spec"]["description"],
                    )
                )

            model_to_use = ChatOpenAI(model=model_to_use_id, **self.openai_kwargs)  # type: ignore

            messages_to_use = body["messages"]

            if is_reasoning_needed == "NO":
                messages_to_use[0]["content"] = messages_to_use[0]["content"] + USER_INTERACTION_PROMPT
                messages_to_use[-1]["content"] = str(messages_to_use[-1]["content"]).replace("#*yes", "").replace("#*no", "").replace("#!!", "").replace("#!", "")

                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": body["messages"]}

                num_tool_calls = 0
                async for event in graph.astream_events(inputs, version="v2", config=config):
                    if num_tool_calls >= 4:
                        break
                    kind = event["event"]
                    data = event["data"]
                    if kind == "on_chat_model_stream":
                        if "chunk" in data and (content := data["chunk"].content):
                            yield content
                    elif kind == "on_tool_start":
                        yield "\n"
                        await send_status(f"Running tool {event['name']}", False)
                    elif kind == "on_tool_end":
                        num_tool_calls += 1
                        await send_status(
                            f"Tool '{event['name']}' returned {data.get('output')}", True
                        )
                        await send_citation(
                            url=f"Tool call {num_tool_calls}",
                            title=event["name"],
                            content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                        )
                return 
            elif is_reasoning_needed == "YES": 
                reasoning_model_id = self.valves.REASONING_MODEL
                if mini_mode == True:
                    reasoning_model_id = self.valves.MINI_REASONING_MODEL

                reasoning_model = ChatOpenAI(model=reasoning_model_id, **self.openai_kwargs)  # type: ignore

                full_content = ""

                reasoning_context = ""

                for msg in body["messages"][:-1]:  # Process all messages except the last one
                    if msg["role"] == "user":
                        if len(msg["content"]) > 400:
                            text_msg = msg["content"][:250] + "\n...(Middle cut)...\n" + msg["content"][-100:]
                        else:
                            text_msg = msg["content"]
                        reasoning_context += f"--- NEXT MESSAGE FROM \"{msg['role'].upper()}\" ---\n{text_msg}"
                    if msg["role"] == "assistant":
                        if len(msg["content"]) > 250:
                            text_msg = msg["content"][:150] + "\n...(Middle cut)...\n" + msg["content"][-50:]
                        else:
                            text_msg = msg["content"]
                        reasoning_context += f"--- NEXT MESSAGE FROM \"{msg['role'].upper()}\" ---\n{text_msg}"

                # Add the last message without cutting it
                last_msg = body["messages"][-1]
                if last_msg["role"] == "user":
                    reasoning_context += f"--- LAST USER MESSAGE/PROMPT ---\n{last_msg['content']}"

                reasoning_context = reasoning_context.replace("#*yes", "").replace("#*no", "").replace("#!!", "").replace("#!", "")

                reasoning_messages = [
                    {
                        "role": "system",
                        "content": REASONING_PROMPT
                    },
                    {
                        "role": "user",
                        "content": reasoning_context
                    }
                ] 

                await send_status(
                        status_message="Reasoning...",
                        done=False,
                    )
                #reasoning_content = reasoning_model.invoke(reasoning_messages, config=config).content
                #assert isinstance(content, str)

                reasoning_bufffer = ""
                update_status = 0
                async for chunk in reasoning_model.astream(reasoning_messages, config=config):
                    content = chunk.content
                    assert isinstance(content, str)
                    reasoning_bufffer += content
                    update_status += 1

                    if update_status >= 5:
                        update_status = 0 
                        await send_status(
                            status_message=f"Reasoning ({len(reasoning_bufffer)})... {reasoning_bufffer[-100:]}",
                            done=False,
                        )

                await send_status(
                            status_message=f"Reasoning ({len(reasoning_bufffer)})... done",
                            done=True,
                        )

                reasoning_content = reasoning_bufffer

                full_content += "<reasoning_agent_output>\n" + reasoning_content + "\n<reasoning_agent_output>"

                await send_citation(
                        url=f"SMART Reasoning",
                        title="SMART Reasoning",
                        content=f"{reasoning_content=}",
                    )

                # Try to find <ask_tool_agent> ... </ask_tool_agent> using re
                # If found, then ask the tool agent
                tool_agent_content = re.findall(r"<ask_tool_agent>(.*?)</ask_tool_agent>", reasoning_content, re.DOTALL)
                print(f"{tool_agent_content=}")

                if len(tool_agent_content) > 0:
                    await send_status(f"Running tool-agent...", False)
                    tool_message = [
                        {
                            "role": "system",
                            "content": TOOL_PROMPT
                        },
                        {
                            "role": "user",
                            "content": "<reasoning_agent_requests>\n" + str(tool_agent_content) + "\n</reasoning_agent_requests>"
                        }
                    ]

                    if not __tools__:
                        tool_agent_response = "Tool agent could not use any tools because the user did not enable any."
                    else:
                        graph = create_react_agent(large_model, tools=tools)
                        inputs = {"messages": tool_message}
                        message_buffer = ""
                        num_tool_calls = 0
                        async for event in graph.astream_events(inputs, version="v2", config=config):  # type: ignore
                            if num_tool_calls > 3:
                                break
                            kind = event["event"]
                            data = event["data"]
                            if kind == "on_chat_model_stream":
                                if "chunk" in data and (content := data["chunk"].content):
                                    message_buffer = message_buffer + content
                            elif kind == "on_tool_start":
                                message_buffer = message_buffer + "\n"
                                await send_status(f"Running tool {event['name']}", False)
                            elif kind == "on_tool_end":
                                num_tool_calls += 1
                                await send_status(
                                    f"Tool '{event['name']}' returned {data.get('output')}", True
                                )
                                await send_citation(
                                    url=f"Tool call {num_tool_calls}",
                                    title=event["name"],
                                    content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                                )

                        tool_agent_response = message_buffer

                    print("TOOL AGENT RESPONSE:\n\n" + str(tool_agent_response))
                    await send_citation(
                        url=f"SMART Tool-use",
                        title="SMART Tool-use",
                        content=f"{tool_agent_response=}",
                    )

                    full_content += "\n\n\n<tool_agent_output>\n" + tool_agent_response + "\n<tool_agent_output>" 

                await send_status(
                        status_message="Reasoning complete.",
                        done=True,
                    )
                
                messages_to_use[-1]["content"] = "<user_input>\n" + messages_to_use[-1]["content"] + "\n</user_input>\n\n" + full_content 
                messages_to_use[0]["content"] = messages_to_use[0]["content"] + USER_INTERACTION_PROMPT
                #messages_to_use[-1]["content"] = messages_to_use[-1]["content"] + "\n\n<preprompt>" + next_agent_preprompt + "</preprompt>"

                graph = create_react_agent(model_to_use, tools=tools)
                inputs = {"messages": messages_to_use}

                num_tool_calls = 0
                async for event in graph.astream_events(inputs, version="v2", config=config):
                    if num_tool_calls >= 4:
                        break
                    num_tool_calls += 1
                    kind = event["event"]
                    data = event["data"]
                    if kind == "on_chat_model_stream":
                        if "chunk" in data and (content := data["chunk"].content):
                            yield content
                    elif kind == "on_tool_start":
                        yield "\n"
                        await send_status(f"Running tool {event['name']}", False)
                    elif kind == "on_tool_end":
                        num_tool_calls += 1
                        await send_status(
                            f"Tool '{event['name']}' returned {data.get('output')}", True
                        )
                        await send_citation(
                            url=f"Tool call {num_tool_calls}",
                            title=event["name"],
                            content=f"Tool '{event['name']}' with inputs {data.get('input')} returned {data.get('output')}",
                        )
                return

            else:
                yield "Error: is_reasoning_needed is not YES or NO"
                return
        except Exception as e:
            yield "Error: " + str(e)
            return