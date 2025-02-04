
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
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
            context=[research_task]
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
            context=[research_task, analysis_task]
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

def main():
    st.set_page_config(page_title="Ghana Education Research Proposal Generator", page_icon="📚")
    
    st.title("📚 Ghana Basic Education Research Proposal Generator")
    st.markdown("Create action research proposals for teacher trainees (Basic School focus)")
    
    with st.sidebar:
        st.header("Configuration")
        api_key = os.getenv("GEMINI_API_KEY")
        serper_key = os.getenv("SERPER_API_KEY")
        if api_key != "":
            st.success("GEMINI_API_KEY Active")
        if serper_key != "":
            st.success("SERPER_API_KEY Active")
    topic = st.text_input(
        "Enter research focus (e.g., 'Improving Reading Comprehension in Class 3'):",
        placeholder="Using Role play method to enhance participation among Basic 5 learners..."
    )
    
    if st.button("Generate Proposal"):
        if not topic:
            st.error("Please enter a research focus topic!")
            return
            
        if not api_key:
            st.error("Please provide your Gemini API key!")
            return
            
        try:
            with st.spinner("Generating research proposal... This may take 2-3 minutes"):
                proposal_crew = ResearchProposalCrew(topic)
                result = proposal_crew.run()
                
                st.success("Proposal Generated Successfully!")
                st.divider()
                
                st.header("Research Proposal")
                st.markdown(result)
                
                st.download_button(
                    label="Download Proposal",
                    data=f"{result}",
                    file_name=f"Proposal_{topic.replace(' ','_')}.md",
                    mime="text/markdown"
                )
                
        except Exception as e:
            st.error(f"Error generating proposal: {str(e)}")
            if "GEMINI_API_KEY" in str(e):
                st.info("Please ensure you have a valid Gemini API key")

if __name__ == "__main__":
    main()
