- We want to create a chat bot with a Hybrid RAG (BM25 + FAISS) and a graphRAG. 
- It will ALWAYS use an external service for LLM needs.
- This is a multi user system and there are 2 roles: user and admin.
- A user can start chats, view their chat history, resume older chats.
- All users should have a profile where they can download their chat history or delete their complete chat history. The should also be able to change and save their preference for light or dark mode.
- When the user starts a new chat we should prompt them with a chat like "Hey this is Doogie! How can I assist you today?"
- We should have a little information icon below a chat that when you mouse over it tells you how many tokens and token/s and other useful statistics about the lat prompt.
- The user should be able to provie a thumbs up or thumbs down on a chat. If they choose thumbs down they should be able to enter a description of what they think is incorrect.
- Since a resoning model could be used we need to deal with <think></think> tags.
- There should be an adminstrative section that only users with admin role can access.
- The admin section should have a couple of sections:

    1. User Management: When a new user registers it requires an admin to approve their access. Users cannot chat until they have been approved by an admin. Also the admin can select the role of user or admin.
    2. LLM Options: This is where the admin can choose between what service to use for the LLM and what models. We should support remote ollama servers, remote LM Studio servers, OpenAI, Openrouter, Deepseek, and Anthropic. We need to be able to select the chat model and the embedding model. We shoudl also be able to set the default system prompt for chats in this section.
    3. RAG Section: This is where a user can upload documents to the RAG. We should support pdf, microsoft documents, md, rst, tst, json, and jsonl files. We should be able to rebuid the entire rag from the docs with a button. We should be able to regenerate the graphrag. There should be a text input section that allows manual adding of information to the rag.
    4. Chat review: This section allows the admin to review chats that were deemed incorrect. There should be a way to mark a chat as reviewed by an admin so multiple admin are 

- I want to use Python 3.13+ and sqlite3
- The backend should be FastAPI.
- Next.js/React for the front end.
- Try and use the latest versions of dependencies.
- I want to use docker and docker compose.
- All testing will be done inside the docker containers.
- The frontend should proxy everything to the backend. That way the UI is just accesing the API.
- The first time the app comes up it shoud create the default admin user.
- I would like a modern web interface to be dynamic and suitable for a chat bot.
- The web interface should default to dark mode but also have a light mode.
- usernames will be email address. 
- We should be able to stream results to the front end so the user doesn't have to wait to start seeing results.
