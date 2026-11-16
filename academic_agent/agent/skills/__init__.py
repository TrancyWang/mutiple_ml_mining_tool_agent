"""可复用的 Agent 工作流技能。"""

from academic_agent.agent.skills.catalog import AgentSkill, skill_catalog
from academic_agent.agent.skills.manager import InstalledSkill, SkillManager, skill_manager

__all__ = ["AgentSkill", "skill_catalog", "InstalledSkill", "SkillManager", "skill_manager"]
