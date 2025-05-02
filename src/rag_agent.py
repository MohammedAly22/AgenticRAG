"""
Module for building a ReAct-style RAG (Retrieval-Augmented Generation) agent
that answers questions based on the 2025 AI Index Report using LangChain.
"""

from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor


from langchain.memory import ConversationBufferMemory


class ReActRagAgent:
    """
    ReAct-style RAG agent for answering questions about the 2025 AI Index Report.

    This agent uses a language model to retrieve relevant passages from a vector store
    and respond with contextually grounded answers, while maintaining conversation history.

    Attributes:
        - llm: A LangChain-compatible LLM for generating responses.
        - memory (ConversationBufferMemory): Stores chat history for context retention.
        - retriever: A retriever object to fetch relevant documents.
        - tools (list): List of tools used by the agent (primarily the retriever).
        - agent: LangChain agent created using the ReAct prompt and retriever tool.
    """

    def __init__(self, llm, vector_store, number_of_retrieved_documents: int = 5):
        """
        Initialize the ReActRagAgent.

        Args:
            - llm: The language model to use for generating responses.
            - vector_store: A LangChain-compatible vector store with embedded documents.
            - number_of_retrieved_documents (int): Number of documents to retrieve per query.
        """

        self.system_template = """
        You're a helpful question-answering assistant named "2025 AI Index Report Assistant" 
        answering questions based on some retrieved excerpts relevant to a user-asked query. 
        You have knowledge regarding the 2025 AI Index Report. 

        When using retrieved excerpts:
          - Enclose excerpts in quotes.
          - Cite page numbers (p. X).
          - No hallucinations.

        Plan → Retrieve → Answer.
        """

        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_template),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                HumanMessagePromptTemplate.from_template("{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        self.llm = llm
        self.memory = ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )

        self.retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": number_of_retrieved_documents,
                "fetch_k": number_of_retrieved_documents * 4,
            },
        )

        tool = create_retriever_tool(
            self.retriever,
            name="ai_index_2025_retriever",
            description="Searches and returns excerpts from the AI Index Report 2025 "
            "related to user query.",
        )

        self.tools = [tool]
        self.agent = create_tool_calling_agent(
            llm=llm, tools=self.tools, prompt=self.prompt
        )

    def create_react_agent_executor(self):
        """
        Create and return the ReAct agent executor, which handles user interactions.

        Returns:
            - AgentExecutor: The configured LangChain agent executor.
        """

        self.react_agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            memory=self.memory,
        )

        return self.react_agent_executor
