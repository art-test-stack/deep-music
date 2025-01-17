'use client'
import { useState } from "react";
import axios from "axios";
import ReactPlayer from "react-player";


export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [songUrl, setSongUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleGenerate = async () => {
    console.log("fetching prompt:", prompt)
    if (!prompt) return;
    setLoading(true);
    setError("");
    setSongUrl("");

    try{
        const response = await fetch("http://localhost:8000/generate", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ prompt: prompt }),
          })
        ;
        if (response) {
            const res = await response.json();
            setSongUrl(res.url);
        } 
    } catch (err) {
        setError("Failed to generate song. Please try again.");
    } finally {
        setLoading(false);
    }};

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>AI Song Creator</h1>
      <p>Enter a prompt to generate your custom song:</p>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        // rows="4"
        placeholder="Write your song prompt here..."
        style={{ width: "100%", padding: "10px", fontSize: "16px" }}
      />
      <button
        onClick={handleGenerate}
        disabled={loading}
        style={{
          marginTop: "10px",
          padding: "10px 20px",
          fontSize: "16px",
          backgroundColor: "#0070f3",
          color: "white",
          border: "none",
          borderRadius: "5px",
          cursor: "pointer",
        }}
      >
        {loading ? "Generating..." : "Generate Song"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {songUrl && (
        <div style={{ marginTop: "20px" }}>
          <h2>Your Song:</h2>
          <ReactPlayer url={songUrl} controls width="100%" height="50px" />
          <a
            href={songUrl}
            download
            style={{
              display: "inline-block",
              marginTop: "10px",
              color: "#0070f3",
              textDecoration: "underline",
            }}
          >
            Download MP3
          </a>
        </div>
      )}
    </div>
  );
}
