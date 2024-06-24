# AirBNB 2024 Q1 10k Financial Analyst Bot 

### Key Features: 
- LLM: `Meta-Llama-3-8B-Instruct`
- Embedding Model: `snowflake-arctic-embed-m`
- PDF Parser: `Llama Index Markdown Mode`
- PDF Parser Instructions: 
```
The provided document is a quarterly report filed by AirBNB,
with the Securities and Exchange Commission (SEC).
Some of the pages include detailed financial information about the company's performance for a specific quarter.
It includes unaudited financial statements, management discussion and analysis, and other relevant disclosures required by the SEC.
It contains many tables.
Try to be precise while answering the questions
```

- Infrastructure: `LangChain` 
- Vector Store: `QDrant`
- LLM Instructions: 
```
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.
```