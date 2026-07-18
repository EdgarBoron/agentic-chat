from contextlib import AsyncExitStack

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState as _BaseAgentState

from app.agent.prompts import get_system_prompt
from app.agent.tools.history_search import search_prompt_history
from app.agent.tools.reference_search import search_prompt_reference
from app.agent.tools.web_search import web_search
from app.config import Settings

TOOLS = [
    web_search,
    search_prompt_reference,
    search_prompt_history,
]


class AgentState(_BaseAgentState):
    target_mode: str


def build_prompt(state: AgentState) -> list:
    return [SystemMessage(content=get_system_prompt(state.get("target_mode")))] + state[
        "messages"
    ]


class AgentHandle:
    """Owns the agent graph and the lifetime of its checkpointer connection."""

    def __init__(self) -> None:
        self.graph = None
        self.llm: ChatOpenAI | None = None
        self._stack = AsyncExitStack()

    async def start(self, settings: Settings) -> None:
        self.llm = ChatOpenAI(
            base_url=settings.vllm_base_url,
            api_key="not-needed",
            model=settings.vllm_model_name,
            temperature=0.4,
            streaming=True,
        )
        saver = await self._stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(settings.checkpoint_db_path)
        )
        self.graph = create_react_agent(
            self.llm,
            TOOLS,
            prompt=build_prompt,
            state_schema=AgentState,
            checkpointer=saver,
        )

    async def stop(self) -> None:
        await self._stack.aclose()


agent_handle = AgentHandle()
