from crewai import Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()

from crewai import LLM

# For Gemini 1.5 Flash (recommended for speed and cost-effectiveness)
llm = LLM(
    model="gemini/gemini-1.5-flash",
    temperature=0.7,  # Adjust creativity (0.0-1.0)
    max_tokens=4000,  # Control response length
    api_key=os.getenv("GEMINI_API_KEY")
)





# Load agents and tasks from YAML (or define inline)
researcher = Agent(config="agents.yaml#researcher")
analyst = Agent(config="agents.yaml#analyst")
writer = Agent(config="agents.yaml#proposal_writer")

research_task = Task(config="tasks.yaml#research_task")
analysis_task = Task(config="tasks.yaml#analysis_task")
proposal_task = Task(config="tasks.yaml#proposal_task")

# Create the crew
proposal_crew = Crew(
  agents=[researcher, analyst, writer],
  tasks=[research_task, analysis_task, proposal_task],
  process=Process.sequential,  # Tasks run in order
  verbose=True
)

# Execute with dynamic topic input
result = proposal_crew.kickoff(inputs={"topic": "AI Ethics"})
print(result)
