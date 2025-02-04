import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class ResearchProposalCrew:
    def __init__(self, topic):
        self.topic = topic
        self.gemini_llm = LLM(
            model="gemini/gemini-2.0-flash-exp",
            temperature=0.5
        )
        
        # Initialize tools
        self.search_tool = SerperDevTool() if os.getenv("SERPER_API_KEY") else None
        
        # Initialize agents
        self.researcher = self.create_researcher()
        self.analyst = self.create_analyst()
        self.writer = self.create_writer()

    def create_researcher(self):
        return Agent(
            role="Senior Research Specialist",
            goal=f"Investigate and gather information about {self.topic}",
            backstory="""An expert researcher with 15 years experience in academic 
            and industrial research. Known for thoroughness and accuracy.""",
            tools=[self.search_tool] if self.search_tool else [],
            llm=self.gemini_llm,
            verbose=True
        )

    def create_analyst(self):
        return Agent(
            role="Lead Data Analyst",
            goal=f"Identify research gaps and opportunities in {self.topic}",
            backstory="""A data scientist specializing in trend analysis and 
            research gap identification.""",
            llm=self.gemini_llm,
            verbose=True
        )

    def create_writer(self):
        return Agent(
            role="Senior Research Proposal Writer",
            goal=f"Write compelling research proposals about {self.topic}",
            backstory="""A professional academic writer with 100+ successful 
            grant proposals to major funding agencies.""",
            llm=self.gemini_llm,
            verbose=True
        )

    def create_tasks(self):
        research_task = Task(
            description=f"""Conduct comprehensive research on {self.topic}.
            Gather information from academic papers, industry reports, 
            and news articles.""",
            expected_output="A 1000-word research report with citations",
            agent=self.researcher,
            async_execution=True
        )

        analysis_task = Task(
            description=f"""Analyze the research data to identify:
            1. Key challenges in {self.topic}
            2. Underexplored research opportunities
            3. Potential societal impacts""",
            expected_output="A structured analysis report with bullet points",
            agent=self.analyst,
            context=[research_task]
        )

        proposal_task = Task(
            description=f"""Write a formal research proposal about {self.topic}
            Include these sections:
            1. Introduction and Background
            2. Research Objectives
            3. Methodology
            4. Expected Outcomes
            5. Budget Estimation
            6. Ethical Considerations""",
            expected_output="A 2000-word research proposal in Markdown format",
            agent=self.writer,
            context=[analysis_task]
        )

        return [research_task, analysis_task, proposal_task]

    def run(self):
        tasks = self.create_tasks()
        crew = Crew(
            agents=[self.researcher, self.analyst, self.writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        return crew.kickoff()

if __name__ == "__main__":
    try:
        topic = input("Enter research topic: ")
        crew = ResearchProposalCrew(topic)
        result = crew.run()
        
        print("\n\n=== FINAL RESEARCH PROPOSAL ===")
        print(result)
        
        # Save to file
        with open(f"{topic.replace(' ', '_')}_proposal.md", "w") as f:
            f.write(result.output) 
        print(f"\nProposal saved to {topic.replace(' ', '_')}_proposal.md")
        
    except Exception as e:
        print(f"Error: {e}")
        if "GEMINI_API_KEY" in str(e):
            print("Please ensure you have a valid Gemini API key in .env file")
