import { useEffect, useState } from "react";

const MAX_ARTICLE_LENGTH = 20_000;

export default function App() {
  const [article, setArticle] = useState("");
  const [health, setHealth] = useState("checking");
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let active = true;

    fetch("/api/v1/health")
      .then((response) => {
        if (!response.ok) {
          throw new Error("The prediction service is unavailable.");
        }
        return response.json();
      })
      .then((data) => {
        if (active) {
          setHealth(data.prediction_ready ? "ready" : "degraded");
        }
      })
      .catch(() => {
        if (active) {
          setHealth("offline");
        }
      });

    return () => {
      active = false;
    };
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setPredictions(null);
    setLoading(true);

    try {
      const response = await fetch("/api/v1/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: article }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "Could not analyze this article.");
      }
      setPredictions(data.predictions);
    } catch (requestError) {
      setError(requestError.message || "Could not connect to the prediction API.");
    } finally {
      setLoading(false);
    }
  }

  const canSubmit = article.trim().length > 0 && !loading;

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="News Signal home">
          <span className="brand-mark" aria-hidden="true">N</span>
          <span>News Signal</span>
        </a>
        <span className={`service-status status-${health}`}>
          <span className="status-dot" />
          {health === "ready" && "Models ready"}
          {health === "checking" && "Checking service"}
          {health === "degraded" && "Models unavailable"}
          {health === "offline" && "API offline"}
        </span>
      </header>

      <section className="hero">
        <p className="eyebrow">AI-ASSISTED ARTICLE ANALYSIS</p>
        <h1>Look closer at the story.</h1>
        <p className="hero-copy">
          Compare two machine-learning models on a news article. Their output is
          a signal to investigate, not proof that a story is true or false.
        </p>
      </section>

      <section className="analyzer-card" aria-labelledby="analyzer-title">
        <div className="card-heading">
          <div>
            <p className="eyebrow">NEW ANALYSIS</p>
            <h2 id="analyzer-title">Paste an article to get started</h2>
          </div>
          <span className="model-count">2 models</span>
        </div>

        {health === "degraded" && (
          <p className="notice" role="status">
            The API is running, but its trained model files are missing or
            incomplete. Restore the model artifacts before requesting results.
          </p>
        )}
        {health === "offline" && (
          <p className="notice" role="status">
            The API is not responding. Start the backend to analyze articles.
          </p>
        )}

        <form onSubmit={handleSubmit}>
          <label htmlFor="article">Article text</label>
          <textarea
            id="article"
            name="article"
            maxLength={MAX_ARTICLE_LENGTH}
            onChange={(event) => setArticle(event.target.value)}
            placeholder="Paste the article or the passage you want to examine..."
            required
            rows={9}
            value={article}
          />
          <div className="form-footer">
            <span className="character-count">
              {article.length.toLocaleString()} / {MAX_ARTICLE_LENGTH.toLocaleString()}
            </span>
            <button disabled={!canSubmit} type="submit">
              {loading ? "Analyzing..." : "Analyze article"}
              {!loading && <span aria-hidden="true">→</span>}
            </button>
          </div>
        </form>

        {error && <p className="error-message" role="alert">{error}</p>}

        {predictions && (
          <section className="results" aria-live="polite">
            <div className="results-heading">
              <div>
                <p className="eyebrow">ANALYSIS COMPLETE</p>
                <h2>Model predictions</h2>
              </div>
            </div>
            <div className="result-grid">
              {Object.entries(predictions).map(([model, label]) => (
                <article className="result-card" key={model}>
                  <span className="result-model">{model}</span>
                  <strong className="result-label">{label}</strong>
                </article>
              ))}
            </div>
            <p className="disclaimer">
              Models can be wrong or disagree. Verify claims with reliable
              sources before drawing conclusions.
            </p>
          </section>
        )}
      </section>

      <footer>
        <span>News Signal</span>
        <span>Machine learning can assist research, not replace verification.</span>
      </footer>
    </main>
  );
}
