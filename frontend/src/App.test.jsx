import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App.jsx";

function jsonResponse(data, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => data,
  };
}

describe("article analysis", () => {
  it("announces uncertainty when the models disagree", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(jsonResponse({ prediction_ready: true }))
        .mockResolvedValueOnce(
          jsonResponse({
            predictions: {
              "Logistic Regression": "Real News",
              BERT: "Fake News",
            },
          }),
        ),
    );

    render(<App />);
    const articleInput = screen.getByRole("textbox", { name: "Article text" });
    expect(articleInput).toHaveAttribute("aria-describedby", "article-help");
    fireEvent.change(articleInput, { target: { value: "A sample article." } });
    fireEvent.click(screen.getByRole("button", { name: "Analyze article" }));

    expect(
      await screen.findByText(/The models disagree\. Treat this result as uncertain/),
    ).toBeInTheDocument();
    expect(screen.getByText("Real News")).toBeInTheDocument();
    expect(screen.getByText("Fake News")).toBeInTheDocument();
    expect(
      screen.getByText(/Your article is sent to the prediction API and is not saved/),
    ).toBeInTheDocument();
  });

  it("does not imply certainty from model agreement", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(jsonResponse({ prediction_ready: true }))
        .mockResolvedValueOnce(
          jsonResponse({
            predictions: {
              "Logistic Regression": "Real News",
              BERT: "Real News",
            },
          }),
        ),
    );

    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Article text" }), {
      target: { value: "A sample article." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyze article" }));

    expect(
      await screen.findByText(/Agreement is not proof that the article is true or false/),
    ).toBeInTheDocument();
  });

  it("renders API validation errors as readable text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn()
        .mockResolvedValueOnce(jsonResponse({ prediction_ready: true }))
        .mockResolvedValueOnce(
          jsonResponse(
            { detail: [{ msg: "Article text must not be blank." }] },
            422,
          ),
        ),
    );

    render(<App />);
    fireEvent.change(screen.getByRole("textbox", { name: "Article text" }), {
      target: { value: "A sample article." },
    });
    fireEvent.click(screen.getByRole("button", { name: "Analyze article" }));

    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Article text must not be blank.",
      );
    });
    expect(screen.getByRole("textbox", { name: "Article text" })).toHaveAttribute(
      "aria-invalid",
      "true",
    );
  });
});
