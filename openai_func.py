import openai
import numpy as np
import copy
import ast
import re
import math
import time
import anthropic
from mistralai import Mistral
# Updated import for newer mistralai version
# from mistralai.models.chat_completion import ChatMessage

# Import API keys from config file
try:
    from config import CLAUDE_API_KEY, MIXTRAL_API_KEY
    claude_api_key_name = CLAUDE_API_KEY
    mixtral_api_key_name = MIXTRAL_API_KEY
except ImportError:
    # Fallback to environment variables if config.py doesn't exist
    import os
    claude_api_key_name = os.environ.get('CLAUDE_API_KEY', '')
    mixtral_api_key_name = os.environ.get('MIXTRAL_API_KEY', '')

def GPT_response(messages, model_name):
  if model_name in ['gpt-4-turbo-preview','gpt-4-1106-preview', 'gpt-4', 'gpt-4o', 'gpt-4-32k', 'gpt-3.5-turbo-0301', 'gpt-4-0613', 'gpt-4-32k-0613', 'gpt-3.5-turbo-16k-0613', 'gpt-3.5-turbo']:
    #print(f'-------------------Model name: {model_name}-------------------')
    response = openai.ChatCompletion.create(
      model=model_name,
      messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": messages}
        ],
      temperature = 0.0,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )
    
  return response.choices[0].message.content

def Claude_response(messages):
  client = anthropic.Anthropic(
    api_key=claude_api_key_name,
  )
  message = client.messages.create(
    model="claude-3-opus-20240229", # claude-3-sonnet-20240229, claude-3-opus-20240229, claude-3-haiku-20240307
    max_tokens=4096,
    temperature=0.0,
    system="",
    messages=[
        {"role": "user", "content": messages}
    ]
  )
  return message.content[0].text

def Mixtral_response(messages, mode = 'normal'):
  model = 'mistral-large-latest'
  client = Mistral(api_key=mixtral_api_key_name)

  if mode == 'json':
    messages_list = [
        {"role": "system", "content": "You are a helpful code assistant. Your task is to generate a valid JSON object based on the given information. Please only produce the JSON output and avoid explaining."}, 
        {"role": "user", "content": messages}
    ]
  elif mode == 'code':
    messages_list = [
    {"role": "system", "content": "You are a helpful code assistant that help with writing Python code for a user requests. Please only produce the function and avoid explaining. Do not add \ in front of _"},
    {"role": "user", "content": messages}
  ]
  else: 
    messages_list = [
    {"role": "user", "content": messages}
  ]

  # No streaming
  chat_response = client.chat.complete(
      model=model,
      messages=messages_list,
      temperature=0.0,
  )
  # import pdb; pdb.set_trace()
  return chat_response.choices[0].message.content

