import { useState } from "react";

interface Props {
  onSubmit: (text: string) => void;
  loading: boolean;
}

export function TextUpload({ onSubmit, loading }: Props) {
  const [text, setText] = useState("");

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setText(reader.result as string);
    reader.readAsText(file);
  };

  return (
    <div style={{ marginBottom: 24 }}>
      <div style={{ marginBottom: 12 }}>
        <label>
          Загрузить файл:{" "}
          <input type="file" accept=".txt,.md" onChange={handleFileUpload} />
        </label>
      </div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Или вставьте текст сюда..."
        rows={10}
        style={{ width: "100%", fontFamily: "monospace" }}
      />
      <button
        onClick={() => onSubmit(text)}
        disabled={loading || !text.trim()}
        style={{ marginTop: 8 }}
      >
        {loading ? "Анализ..." : "Анализировать"}
      </button>
    </div>
  );
}
