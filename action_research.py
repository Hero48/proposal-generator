import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

load_dotenv()

class ResearchProposalCrew:
    def __init__(self, topic):
        self.topic = topic
        self.llm = LLM(
            model="gemini/gemini-2.0-flash-exp",
            temperature=0.5
        )
       
        self.researcher = Agent(
            role="Ghana Education Analyst",
            goal="Identify key issues in basic schools",
            backstory="Expert in Ghana's primary education system",
            tools=[SerperDevTool()] if os.getenv("SERPER_API_KEY") else [],
            llm=self.llm,
            verbose=True
        )
        
        self.designer = Agent(
            role="Action Research Specialist",
            goal="Create practical interventions for teacher trainees",
            backstory="Experienced in classroom action research design",
            llm=self.llm,
            verbose=True
        )
        
        self.writer = Agent(
            role="Proposal Architect",
            goal="Structure proposals using academic guidelines",
            backstory="Skilled in Ghanaian research formatting",
            llm=self.llm,
            verbose=True
        )

    def create_tasks(self):
        research_task = Task(
            description=f"""Investigate {self.topic} in Ghanaian basic schools.
            Focus on:
            - Current MOE/GES policies
            - Common classroom challenges
            - Relevant local case studies""",
            expected_output="Bullet-point research summary with citations",
            agent=self.researcher
        )

        analysis_task = Task(
            description=f"""Design action research plan for {self.topic}:
            1. Define measurable objectives
            2. Create 4-week intervention plan
            3. Suggest assessment methods
            4. Outline ethical considerations""",
            expected_output="Structured action plan with timeline",
            agent=self.designer,
            context=[research_task]  # Fixed dependency
        )

        proposal_task = Task(
            description=f"""Compile formal research proposal using this structure:
            
            RESEARCH PROPOSAL: [TOPIC]
            
            1. INTRODUCTION
            1.1 Background (Ghanaian education context)
            1.2 Problem Statement
            1.3 Purpose & Objectives
            
            2. RESEARCH DESIGN
            2.1 Action Research Approach
            2.2 Target Group (Grade/Subject/School Type)
            2.3 Data Collection Methods
            2.4 Intervention Strategy
            
            3. ETHICAL CONSIDERATIONS
            3.1 Participant Consent
            3.2 Data Privacy Measures
            
            4. EXPECTED OUTCOMES
            4.1 Anticipated Impacts
            4.2 Relevance to Teacher Training
            
            5. WORK PLAN
            5.1 8-Week Timeline
            5.2 Resource Requirements
            
            References (APA format)""",
            expected_output="Full proposal document in Markdown",
            agent=self.writer,
            context=[research_task, analysis_task]  # Correct task dependencies
        )

        return [research_task, analysis_task, proposal_task]

    def run(self):
        crew = Crew(
            agents=[self.researcher, self.designer, self.writer],
            tasks=self.create_tasks(),
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff()

if __name__ == "__main__":
    try:
        topic = input("Enter research focus (e.g., 'Improving Reading Comprehension in Class 3'): ")
        proposal_crew = ResearchProposalCrew(topic)
        result = proposal_crew.run()
        
        print("\n\n=== ACTION RESEARCH PROPOSAL ===")
        print(result.output)
        
        filename = f"Proposal_{topic.replace(' ','_')}.md"
        with open(filename, "w") as f:
            f.write(result.output)
        print(f"\nSaved to {filename}")
        
    except Exception as e:
        print(f"Error: {e}")

