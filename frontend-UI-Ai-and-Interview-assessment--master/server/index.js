require("dotenv").config();

const express = require("express");
const cors = require("cors");
const axios = require("axios");

const db = require("./db");
const { router: authRouter } = require("./auth");

const app = express();
const PORT = process.env.PORT || 5000;

/* ---------------- Middleware ---------------- */

app.use(cors());
app.use(express.json());

/* ---------------- Routes ---------------- */

// Auth routes
app.use("/api/auth", authRouter);

// Test route
app.get("/", (req, res) => {
  res.send("Backend Server Running");
});

/* ---------------- LLM Integration ---------------- */

app.post("/api/evaluate", async (req, res) => {
  try {
    const { question, answer } = req.body;

    if (!question || !answer) {
      return res.status(400).json({
        error: "Question and answer are required",
      });
    }

    // Call FastAPI LLM server
    const response = await axios.post(
      "http://127.0.0.1:8000/evaluate",
      {
        question: question,
        answer: answer,
      }
    );

    res.json(response.data);

  } catch (error) {

    console.error(
      "LLM API Error:",
      error.response?.data || error.message
    );

    res.status(500).json({
      error: "Failed to connect to LLM service",
    });
  }
});

/* ---------------- Start Server ---------------- */

const startServer = async () => {
  try {

    await db.initDb();

    app.listen(PORT, () => {
      console.log(`Server running on http://localhost:${PORT}`);
    });

  } catch (error) {
    console.error("Error initializing database:", error);
  }
};

startServer();