
  AI Interview Evaluation System

Overview

The AI Interview Evaluation System is a full-stack application designed to simulate and evaluate interview responses using Artificial Intelligence.
The system allows users to submit answers to interview questions and receive AI-generated evaluation scores and feedback.

This project integrates multiple technologies including React (frontend), Node.js (backend), and FastAPI with an LLM evaluation engine.

---

System Architecture

Frontend (React + Vite)
↓
Node.js Backend (Express)
↓
FastAPI LLM Evaluation Service
↓
AI-Generated Scoring and Feedback

---

Technologies Used

Frontend

- React
- Vite
- JavaScript
- HTML/CSS

Backend

- Node.js
- Express.js
- Axios (for API communication)

AI Evaluation Service

- FastAPI
- Python
- LLM-based evaluation engine

Database

- PostgreSQL

---

Features

- AI-powered interview answer evaluation
- Backend integration with FastAPI LLM service
- Real-time scoring and feedback
- Modular backend architecture
- API-based communication between services

---

API Integration Flow

Client / Frontend
→ Backend API ("/api/evaluate")
→ FastAPI LLM Endpoint ("/evaluate")
→ AI evaluates response
→ Backend returns structured scoring and feedback

---

Example API Request

POST "/api/evaluate"

Request Body:

{
"question": "What is Artificial Intelligence?",
"answer": "Artificial intelligence is the simulation of human intelligence by machines."
}

Example Response:

{
"relevance_score": 5.3,
"clarity_score": 4.8,
"technical_accuracy": 5.3,
"communication_score": 4.5,
"overall_score": 4,
"feedback": "The answer demonstrates intermediate-level understanding with room for improvement."
}

---

How to Run the Project

1. Start Backend Server

Navigate to the server folder:

npm run dev

Backend will run at:

http://localhost:5000

---

2. Start Frontend

npm run dev

Frontend will run at:

http://localhost:5173

---

3. Start LLM FastAPI Service

python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

FastAPI will run at:

http://127.0.0.1:8000/docs

---

Project Purpose

This project demonstrates backend module integration, API communication, and AI-powered evaluation systems, making it suitable for learning full-stack development and AI service integration.

---

Author

Satyam Mishra

Electronics & Communication Engineering
Babu Banarasi Das Institute of Technology and Management, Lucknow

---