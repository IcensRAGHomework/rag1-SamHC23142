import json
import traceback
import requests
from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import ClassVar


gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

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
    try:
        holidays = json.loads(result)
        if isinstance(holidays, list):
            print(f"Successfully fetched holidays for {month}/{year} in {country}:")
            for holiday in holidays:
                print(f"- {holiday['name']} on {holiday['date']}")
        else:
            print(f"Unexpected result format: {result}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON: {result}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
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
    messages = [HumanMessage(question)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)
    
    for tool_call in ai_msg.tool_calls:
        selected_tool = {"calendarific": CalendarificTool()}[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    
    response = llm_with_tools.invoke(messages)
    
    return response.content

def generate_hw03(question2, question3):
    pass
    
def generate_hw04(question):
    pass
    
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
