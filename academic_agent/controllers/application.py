"""主 Controller：组合会话、项目和 Agent 用例。"""

from __future__ import annotations

from dataclasses import dataclass

from academic_agent.controllers.agent import AgentController
from academic_agent.controllers.auth import AuthController
from academic_agent.controllers.projects import ProjectController
from academic_agent.controllers.sessions import SessionController
from academic_agent.models import ApplicationState
from academic_agent.repositories import SQLiteConversationRepository, WorkspaceProjectRepository


@dataclass(slots=True)
class ApplicationController:
    state: ApplicationState
    sessions: SessionController
    projects: ProjectController
    agent: AgentController
    auth: AuthController

    @classmethod
    def create(cls, state: ApplicationState) -> "ApplicationController":
        conversation_repository = SQLiteConversationRepository()
        project_repository = WorkspaceProjectRepository()
        return cls(
            state=state,
            sessions=SessionController(state, conversation_repository),
            projects=ProjectController(state, project_repository),
            agent=AgentController(state),
            auth=AuthController(),
        )

    def set_mode(self, value: object) -> bool:
        return self.state.set_mode(value)
