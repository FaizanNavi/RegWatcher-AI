import logging
from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, END

from .router import RouterAgent
from .retrieval import RetrievalAgent
from .sql_agent import SQLAgent
from .response_agents import DraftGeneratorAgent, CriticAgent, ValidatorAgent
from ..utils.config import MAX_CRITIC_CYCLES, MIN_CRITIC_SCORE

logger = logging.getLogger(__name__)

class RegWatcherState(TypedDict):
    query: str
    route: str
    retrieved_chunks: List[Dict]
    draft: str
    critic_score: int
    critic_feedback: str
    critic_hallucinations: List[str]
    critic_iterations: int
    is_valid: bool
    validation_issues: List[str]
    final_response: str
    status: str

class RegWatcherOrchestrator:
    def __init__(self):
        self.router = RouterAgent()
        self.retriever = RetrievalAgent()
        self.sql_agent = SQLAgent()
        self.drafter = DraftGeneratorAgent()
        self.critic = CriticAgent()
        self.validator = ValidatorAgent()
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        logger.info("Initializing LangGraph orchestrator")
        workflow = StateGraph(RegWatcherState)
        
        workflow.add_node("route", self.router.route)
        workflow.add_node("retrieve_rag", self.retriever.retrieve)
        workflow.add_node("retrieve_sql", self.sql_agent.query)
        workflow.add_node("draft", self.drafter.generate)
        workflow.add_node("critic", self.critic.review)
        workflow.add_node("validate", self.validator.validate)
        
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            self._route_decision,
            {
                "rag": "retrieve_rag",
                "sql": "retrieve_sql",
                "hybrid": "retrieve_rag" 
            }
        )
        
        workflow.add_edge("retrieve_rag", "draft")
        workflow.add_edge("retrieve_sql", "draft")
        workflow.add_edge("draft", "critic")
        
        workflow.add_conditional_edges(
            "critic",
            self._critic_decision,
            {"rewrite": "draft", "approve": "validate"}
        )
        
        workflow.add_edge("validate", END)
        
        try:
            from langgraph.checkpoint.redis import RedisSaver
            from redis import Redis
            from ..utils.config import REDIS_URL
            redis_conn = Redis.from_url(REDIS_URL)
            self.checkpointer = RedisSaver(redis_conn)
            logger.info("Configured RedisSaver for LangGraph checkpointing")
            return workflow.compile(checkpointer=self.checkpointer)
        except Exception as e:
            logger.warning(f"Failed to configure RedisSaver, using default memory checkpointer: {e}")
            from langgraph.checkpoint.memory import MemorySaver
            return workflow.compile(checkpointer=MemorySaver())

    def _route_decision(self, state: RegWatcherState) -> str:
        route = state.get("route", "rag")
        logger.info(f"Orchestrator routing to: {route}")
        if route == "sql":
            return "sql"
        return "rag"

    def _critic_decision(self, state: RegWatcherState) -> str:
        score = state.get("critic_score", 10)
        iterations = state.get("critic_iterations", 0)
        if score >= MIN_CRITIC_SCORE:
            logger.info("Critic approved draft")
            return "approve"
        if iterations >= MAX_CRITIC_CYCLES:
            logger.info(f"Max critic cycles ({MAX_CRITIC_CYCLES}) reached, auto-approving")
            return "approve"
        logger.info("Critic rejected draft, sending for rewrite")
        return "rewrite"

    def run(self, query: str, thread_id: str = "default_thread") -> Dict[str, Any]:
        initial_state = {
            "query": query,
            "route": "",
            "retrieved_chunks": [],
            "draft": "",
            "critic_score": 0,
            "critic_feedback": "",
            "critic_hallucinations": [],
            "critic_iterations": 0,
            "is_valid": False,
            "validation_issues": [],
            "final_response": "",
            "status": "Starting..."
        }
        logger.info(f"RegWatcher execution started for query: {query} [Thread: {thread_id}]")
        config = {"configurable": {"thread_id": thread_id}}
        return self.workflow.invoke(initial_state, config=config)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    orch = RegWatcherOrchestrator()
    print("RegWatcher Graph compiled!")
