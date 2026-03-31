import { useState } from "react";
import { MentionGraph } from "../graph/MentionGraph";
import { TextUpload } from "../upload/TextUpload";
import type { GraphData } from "../utils/types";

export function App() {
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (text: string) => {
    setLoading(true);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setGraphData(data.graph);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
      <h1>Mention Map</h1>
      <p>NLP-граф упоминаний персонажей в текстах</p>
      <TextUpload onSubmit={handleAnalyze} loading={loading} />
      {graphData && <MentionGraph data={graphData} />}
    </div>
  );
}
