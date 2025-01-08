import json
import traceback

from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


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
    
def generate_hw02(question):
    pass
    
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
