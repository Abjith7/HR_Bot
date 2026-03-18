# rag🤖 AI HR Chatbot with ATS Resume Checker

An AI-powered HR assistant that answers employee queries and evaluates resumes using ATS (Applicant Tracking System) principles.

🚀 Features
💬 HR Assistant

Answers employee queries about policies, leave, benefits, etc.

Context-aware conversational chatbot

Instant responses using AI

📄 ATS Resume Checker

Analyzes resumes for ATS compatibility

Highlights missing keywords and formatting issues

Provides improvement suggestions

Helps optimize resumes for job applications

🛠️ Tech Stack

Frontend: Streamlit

Backend: Python

AI Models: OpenAI / Google Gemini

Architecture: RAG (Retrieval-Augmented Generation)

Libraries: Pandas, dotenv

📂 Project Structure
project/
│
├── app.py
├── backend/
│   ├── agent.py
│   ├── llm_service.py
│   ├── tools.py
│
├── .env.example
├── .gitignore
└── README.md
⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/your-repo.git
cd your-repo
2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate
3. Install dependencies
pip install -r backend/requirements.txt
4. Setup environment variables

Create a .env file:

OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
5. Run the app
streamlit run app.py
💡 Example Queries
HR Bot

“What is the leave policy?”

“Explain employee benefits”

“What is the notice period?”

Resume Checker

Upload resume → get ATS score

“How can I improve my resume?”

“Is my resume ATS-friendly?
