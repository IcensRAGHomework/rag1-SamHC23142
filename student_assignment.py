import re
import json
import traceback
import requests
import base64

from mimetypes import guess_type

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import ClassVar


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

def parse_json_markdown(json_string: str) -> dict:
    match = re.search(r"""```
                          (?:json)?
                          (.*)```""", json_string, flags=re.DOTALL|re.VERBOSE)
    if match is None:
        json_str = json_string
    else:
        json_str = match.group(1)
    json_str = json_str.strip()
    parsed = json.loads(json_str, strict=False)
    return parsed
class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        result = super().encode(obj)
        formatted_result = result.replace('"Result": {', '"Result":\n  {')
        return formatted_result
        
def to_json(response):
    with open('memorial_day_response.json', 'w', encoding='utf-8') as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    return response

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    response_schemas = {
        "name": "get_memorial_day",
        "description": "Get information about a Taiwan memorial day",
        "parameters": {
            "type": "object",
            "properties": {
                "Result": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "date": {
                                "type": "string",
                                "description": "The date of the memorial day in YYYY-MM-DD format",
                            },
                            "name": {
                                "type": "string",
                                "description": "The name of the memorial day in Traditional Chinese",
                            },
                        },
                        "required": ["date", "name"],
                    },
                },
            },
            "required": ["Result"],
        },
    }

    structured_llm = llm.with_structured_output(response_schemas)

        
    examples = [
        {"input": "2024年1月紀念日", "output":
        {
          "Result":[
            {
              "date": "2024-01-01",
              "name": "開國紀念日"
            }
          ]
        }},
        {"input": "2024年2月紀念日", "output": 
        {
          "Result":[
            {
              "date": "2024-02-28",
              "name": "和平紀念日"
            }
          ]
        }},
        {"input": "2024年3月紀念日", "output":
        {
          "Result":[
            {
              "date": "2024-06-10",
              "name": "端午節"
            }
          ]
        }},
    ]
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_prompt,
        examples = examples,
    )
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an assistant that provides information about Taiwan's memorial days.
                        Please provide the details of the most important memorial day in Taiwan that is also a public holiday (in Traditional Chinese)."""),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
        
    chain = final_prompt | structured_llm
    
    response = chain.invoke({"input": question})
    
    formatted_response = json.dumps(response, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)###json.dumps(response, ensure_ascii=False, indent=2)
    
    return formatted_response
CALENDARIFIC_API_KEY = "3lvN3LXLOX9w25I5aChHnrRNE38jzWz8"
class CalendarificInput(BaseModel):
    year: int = Field(..., description="The year for which to find all holidays and observances")
    country: str = Field(..., description="The country code (e.g., 'TW' for Taiwan)")
    month: int = Field(..., description="The month (1-12) for which to find all holidays and observances")
class CalendarificTool(BaseTool):
    name: ClassVar[str] = "Calendarific"
    description: ClassVar[str] = "Use this tool to retrieve all memorial days and holidays for a specific country, year, and month. It returns a comprehensive list of all national holidays, observances, and significant dates for the specified parameters."
    args_schema: ClassVar[type] = CalendarificInput

    def _run(self, year: int, country: str, month: int) -> str:
        base_url = "https://calendarific.com/api/v2/holidays"
        params = {
            "api_key": CALENDARIFIC_API_KEY,
            "country": country,
            "year": year,
            "month": month,
            "type": ""
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            holidays = response.json()['response']['holidays']
            return json.dumps([{"name": h['name'], "date": h['date']['iso']} for h in holidays], ensure_ascii=False)
        else:
            return f"Error fetching holidays: {response.status_code}"

def test_calendarific_tool():
    tool = CalendarificTool()
    year = 2024
    country = "TW"
    month = 10

    result = tool._run(year, country, month)
    print(result)
    # try:
        # holidays = json.loads(result)
        # if isinstance(holidays, list):
            # print(f"Successfully fetched holidays for {month}/{year} in {country}:")
            # for holiday in holidays:
                # print(f"- {holiday['name']} on {holiday['date']}")
        # else:
            # print(f"Unexpected result format: {result}")
    # except json.JSONDecodeError:
        # print(f"Error decoding JSON: {result}")
    # except Exception as e:
        # print(f"An error occurred: {str(e)}")
        
def generate_hw02(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    
    tools = [CalendarificTool()]
    llm_with_tools = llm.bind_tools(tools)
    response_schemas = [
        ResponseSchema(
            name="Result",
            description="List of holidays",
            type="array",
            properties={
                "date": {"type": "string", "description": "Date of the holiday in YYYY-MM-DD format"},
                "name": {"type": "string", "description": "Name of the holiday in Traditional Chinese"}
            }
        )
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    system_message = SystemMessage(content=f"""You are a competent AI assistant designed to help employees across various departments perform work-related tasks efficiently. Your primary objective is to enhance employee productivity by delivering high-quality, task-specific responses while maintaining a professional and approachable tone.

    When asked about holidays or memorial days in Taiwan, use the Calendarific tool to fetch accurate information. After retrieving the information, format your response as a JSON object with the following structure:

    {output_parser.get_format_instructions()}

    Ensure all holiday names are in Traditional Chinese. Include all relevant holidays for the specified period. Maintain a professional yet approachable tone in any additional explanations.
    Example output format:
    {{
        "Result": [
            {{
                "date": "YYYY-MM-DD",
                "name": "Day Name"
            }},
            {{
                "date": "YYYY-MM-DD",
                "name": "Day Name"
            }},
            ...
        ]
    }}
    Always provide the output in this exact JSON format, without any additional markdown formatting or code block indicators.""")

    messages = [
    system_message,
    HumanMessage(content=question)
]

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    for tool_call in ai_msg.additional_kwargs.get('tool_calls', []):
        selected_tool = {"calendarific": CalendarificTool()}[tool_call["function"]["name"].lower()]
        tool_output = selected_tool.invoke(json.loads(tool_call["function"]["arguments"]))
        messages.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))

    final_response = llm_with_tools.invoke(messages)
    parsed_json = parse_json_markdown(final_response.content)
    parsed_json = json.dumps(parsed_json, ensure_ascii=False, indent=2)
    
    return parsed_json

def generate_hw03(question2, question3):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
        
    tools = [CalendarificTool()]
    
    system_message = """"You are an assistant knowledgeable about every country's festivals.
    following JSON format:
    {{
        "Result": {{
            "add": "This is a boolean value indicating whether the holiday needs to be added to the holiday list. Based on the question, determine if the holiday exists in the list. If it doesn't exist, set to true; otherwise, false",
            "reason": "Describe why the holiday needs to be added or not. Specifically, explain whether the holiday already exists in the list and provide the current content of the list."
        }}
    }}
    Ensure all holiday names are in Traditional Chinese. Maintain a professional yet approachable tone in any additional explanations."""


    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
        
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    
    result2 = generate_hw02(question2)
    chat_history = get_session_history("<foo>")
    chat_history.add_user_message(question2)
    chat_history.add_ai_message(result2)
    
    response = agent_with_chat_history.invoke(
        {"input": question3},
        config={"configurable": {"session_id": "<foo>"}},
    )
    parsed_json = parse_json_markdown(response['output'])
    parsed_json = json.dumps(parsed_json, ensure_ascii=False, indent=2)
    
    return parsed_json

def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"
    
def generate_hw04(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    # Example usage
    image_path = 'baseball.png'
    data_url = local_image_to_data_url(image_path)
    # print("Data URL:", data_url)
    
    system_message = """You are a helpful assistant, you only can output a number"""
    
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=[
            {
                "type": "text",
                "text": question
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": data_url
                }
            }
        ])
    ]

    
    response = llm.invoke(messages)
    
    score = int(response.content.strip())
    
    result = json.dumps({"Result": {"score": score}}, indent=2)
    
    return result
    
def demo(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )
    message = HumanMessage(
            content=[
                {"type": "text", "text": question},
            ]
    )
    response = llm.invoke([message])
    
    return response   
