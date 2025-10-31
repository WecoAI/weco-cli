# Constraints
- Make sure the cost tracking is correct
- The average cost per query should be lower than $0.02 per query, the baseline has around $0.007 per query

# Here's a list of ideas to be tried:
1. Try more recent models like gpt-5, note gpt-5 don't have temperature argument
2. Try response API as below:

<api>
Responses API (concise)

Endpoint:
- POST https://api.openai.com/v1/responses

Purpose:
- Generate text or JSON from text and/or image inputs. Supports optional tools.

Minimum request fields:
- model: string (e.g., gpt-5, gpt-4o)
- input: string | array (text and/or image items)

Useful options:
- max_output_tokens: integer
- temperature or top_p: sampling controls (non-reasoning models)
- reasoning: object (gpt-5, o-series) — controls for reasoning models
- tools: array (web_search, file_search, function calls)
- tool_choice: string | object
- stream: boolean (default false)
- store: boolean (default true)
- parallel_tool_calls: boolean (default true)

Conversation (optional):
- conversation: string | object, or previous_response_id: string

Notes:
- gpt-5 does not support temperature; prefer reasoning options instead.

Example request:

```json
{
  "model": "gpt-5",
  "input": "Explain how to extract a line plot from an image.",
  "max_output_tokens": 400,
  "reasoning": { "effort": "medium" },
  "stream": false
}
```
</api>

3. Try playing with paramters like reasoning efforts
4. Try to build tools for response api to use, or allow python interpretor
5. Try to add preprocessing or post processing pipelines