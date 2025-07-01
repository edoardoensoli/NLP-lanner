# API Keys Configuration

This project requires API keys for various LLM services. Follow these steps to set up your API keys securely.

## Setup Instructions

1. **Copy the example configuration file:**
   ```powershell
   cp config.example.py config.py
   ```

2. **Edit the `config.py` file** and replace the placeholder values with your actual API keys:

   ```python
   # OpenAI API Key (required for GPT models)
   OPENAI_API_KEY = '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'
   
   # Claude API Key (optional, only if using Claude models)
   CLAUDE_API_KEY = 'your_actual_claude_api_key_here'
   
   # Mixtral API Key (optional, only if using Mixtral models)  
   MIXTRAL_API_KEY = 'your_actual_mixtral_api_key_here'
   
   # Google API Key (optional, for some Google services)
   GOOGLE_API_KEY = 'your_actual_google_api_key_here'
   ```

## Where to Get API Keys

- **OpenAI:** https://platform.openai.com/api-keys
- **Claude (Anthropic):** https://console.anthropic.com/
- **Mixtral (Mistral AI):** https://console.mistral.ai/
- **Google:** https://console.cloud.google.com/

## Security Notes

- **Never commit `config.py` to git** - it's already included in `.gitignore`
- Keep your API keys secure and don't share them
- Monitor your API usage to avoid unexpected charges
- Rotate your keys regularly for better security
